"""A block manager that manages token blocks."""
import math
from abc import ABC, abstractmethod
from itertools import count, takewhile
from os.path import commonprefix
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
import copy as cp

logger = init_logger(__name__)


class BlockAllocatorBase(ABC):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    @abstractmethod
    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        pass

    @abstractmethod
    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def free(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_total_blocks(self) -> int:
        pass

    @abstractmethod
    def contains_block(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        pass


class CachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.current_num_blocks = 0
        self.cached_blocks: Dict[int, PhysicalTokenBlock] = {}

        self.evictor: Evictor = make_evictor(eviction_policy)

        self.default_hash_ctr = count()

    def allocate_block(self, block_hash: int,
                       num_hashed_tokens: int) -> PhysicalTokenBlock:
        if self.current_num_blocks == self.num_blocks:
            block = self.evictor.evict()
            block.block_hash = block_hash
            block.num_hashed_tokens = num_hashed_tokens
            return block
        block = PhysicalTokenBlock(device=self.device,
                                   block_number=self.current_num_blocks,
                                   block_size=self.block_size,
                                   block_hash=block_hash,
                                   num_hashed_tokens=num_hashed_tokens)
        self.current_num_blocks += 1
        return block

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if block_hash is None:
            block_hash = next(self.default_hash_ctr)
        if block_hash in self.evictor:
            assert block_hash not in self.cached_blocks
            block = self.evictor.remove(block_hash)
            assert block.ref_count == 0
            self.cached_blocks[block_hash] = block
            block.ref_count += 1
            assert block.block_hash == block_hash
            return block
        if block_hash not in self.cached_blocks:
            self.cached_blocks[block_hash] = self.allocate_block(
                block_hash, num_hashed_tokens)
        block = self.cached_blocks[block_hash]
        assert block.block_hash == block_hash
        block.ref_count += 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            assert block.block_hash not in self.evictor
            self.evictor.add(block)

            # Remove the block from the cached_blocks
            del self.cached_blocks[block.block_hash]

    def get_num_free_blocks(self) -> int:
        return (self.num_blocks - self.current_num_blocks +
                self.evictor.num_blocks)

    def get_num_total_blocks(self) -> int:
        return self.num_blocks

    def contains_block(self, block_hash: int) -> bool:
        return block_hash in self.cached_blocks or block_hash in self.evictor

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        # Update the hash of block and the cached_blocks dictionary.
        assert not self.contains_block(block_hash)
        old_hash = block.block_hash
        block.block_hash = block_hash
        del self.cached_blocks[old_hash]
        self.cached_blocks[block_hash] = block


class UncachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: BlockTable = []
        # Our modification.
        self.free_shared_blocks: BlockTable = []
            
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size,
                                       block_hash=-1,
                                       num_hashed_tokens=0)
            self.free_blocks.append(block)
            
        for i in range(num_blocks, int(2*num_blocks)):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size,
                                       block_hash=-1,
                                       num_hashed_tokens=0)
            self.free_shared_blocks.append(block)

    def allocate_shared(self,) -> PhysicalTokenBlock:
        if not self.free_shared_blocks:
            raise ValueError("Out of memory! No free shared blocks are available.")
        block = self.free_shared_blocks.pop()
        block.ref_count = 1
        return block
            
    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block
    
    def free_shared(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_shared_blocks.append(block)
    
    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)
            

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)
    
    def get_num_free_shared_blocks(self) -> int:
        return len(self.free_shared_blocks)

    def get_num_total_blocks(self) -> int:
        return self.num_blocks

    def contains_block(self, block_hash: int) -> bool:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")


class BlockSpaceManagerV1(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        if enable_caching and sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is not allowed with prefix caching enabled!")

        self.block_sliding_window = None
        if sliding_window is not None:
            # Round up to nearest block size to regularize sliding window
            # allocation sizes.
            self.block_sliding_window = math.ceil(sliding_window / block_size)

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        if self.enable_caching:
            logger.info("Automatic prefix caching is enabled.")
            self.gpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
        else:
            self.gpu_allocator = UncachedBlockAllocator(
                Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator = UncachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

        #shared_block_tables is readable & writable by all the layers.
        self.shared_block_tables: Dict[int, BlockTable] = {}
        # Mapping: req_id -> BlockTable
        # Note that each SequenceGroup has a unique
        # request ID
        self.cross_block_tables: Dict[str, BlockTable] = {}

    def _get_seq_num_required_blocks(self, seq: Sequence) -> int:
        return 0 if seq is None \
            else seq.n_blocks
    
    def can_transform(self, seq_groups):
        #the logic here is to check whether there will be enough blocks, if we transform some existing 
        #kv-cache-utilized decoding requests to hidden-cache-utilized decoding requests.
        #Then the preemption logic would be: if all the existing requests are in hidden-cache-utilized
        #mode, then preempt the lowest priority ones.
        decode_adjust = None
        #Nope the logic can be simpler. Since we would only require one extra block.
        target_seq_group = 0
        transform_flag = False
        num_free_shared_gpu_blocks = self.gpu_allocator.get_num_free_shared_blocks()
        
        for seq_group in seq_groups:
            #for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            #we should count into all the on-going requests.
            seq = seq_group.get_seqs()[0]
            #since the potential blocks have not been assigned, seq.n_blocks may not reflect the actual 
            #number of blocks assigned at the current moment.
            #n_blocks_before_update = len(self.block_tables[seq.seq_id])
            #if seq.use_hidden or n_blocks_before_update == 1:
            if seq.use_hidden or seq.n_blocks == 1:
                target_seq_group += 1
                continue
            else:
                #we also need to check whether there is enough shared KV cache.
                #if num_free_shared_gpu_blocks - n_blocks_before_update >= 0:
                #the reason for seq.n_blocks here is that when we schedule a transformation of 
                #a decode request, we actually still generate a new token in a prefill/recompute way.
                if num_free_shared_gpu_blocks - seq.n_blocks >= 0:
                    transform_flag = True
                    decode_adjust = target_seq_group
                    break
                else:
                    transform_flag = False
                    break
            
        return transform_flag, decode_adjust
                
    def can_allocate_prefill(self, curr_group, seq_groups, backup_groups, \
                             num_curr_batched_tokens, token_budget, num_curr_seqs, max_num_seqs):
        prefill_adjust, decode_adjust = [], []
        num_required_blocks = self._get_seq_num_required_blocks(
            curr_group.get_seqs(status=SequenceStatus.WAITING)[0])
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_free_shared_gpu_blocks = self.gpu_allocator.get_num_free_shared_blocks()
        actual_num_free_gpu_blocks = num_free_gpu_blocks - self.watermark_blocks
        actual_num_free_shared_gpu_blocks = num_free_shared_gpu_blocks - self.watermark_blocks
        
        
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER, prefill_adjust, decode_adjust
        if actual_num_free_gpu_blocks - num_required_blocks >= 0:
            return AllocStatus.OK, prefill_adjust, decode_adjust
        else:
            #IMPORTATNT!!if the current sequence goes into this 'else' and can be scheduled,
            #itself requires to use hidden cache.
            
            #So remember to mark the current request as well when checking free_shared blocks!!!
            actual_num_free_shared_gpu_blocks -= num_required_blocks
            actual_num_free_gpu_blocks -= math.ceil(num_required_blocks/2)
            ##############
            #There should be 4 cases here. 
            #Case 1. actual_num_free_gpu_blocks is OK, actual_num_free_shared_gpu_blocks is not OK.
            #Case 2. actual_num_free_gpu_blocks is not OK, actual_num_free_shared_gpu_blocks is OK.
            #Case 3. actual_num_free_gpu_blocks & actual_num_free_shared_gpu_blocks are both not OK. 
            #Case 4. actual_num_free_gpu_blocks & actual_num_free_shared_gpu_blocks are both OK. 
            
            #Case 1: cannot be scheduled (no space for transformation)
            #Case 2: maybe can be scheduled (by transforming the other candidates)
            #Case 3: cannot be scheduled. (no space for transformation)
            #Case 4: can be scheduled directly (without transforming the other candidates)
            
            #So I think the logic here is to check the shared_blocks first
            if actual_num_free_shared_gpu_blocks < 0: #shared block not enough for the current request.
                #Case 1 & Case 3
                return AllocStatus.LATER, prefill_adjust, decode_adjust
            else: #shared block is enough.
                if actual_num_free_gpu_blocks >= 0: #Case 4
                    return AllocStatus.OK_HIDDEN, prefill_adjust, decode_adjust
                else: #Case 2
                    #For case 2, there are multiple different cases as well.
                    #Case 2.1: return during adjusting previous prefill candidates.
                    #Case 2.1.1: run out of shared blocks. [cannot scheduled]
                    #Case 2.1.2: reach the required blocks. [can be scheduled]
                    #Case 2.2: return during adjusting decode candidate (prerequisite: adjusting all 
                    #previous prefill candidates is not enough.)
                    #Case 2.2.1: run out of shared blocks. [cannot be scheduled]
                    #Case 2.2.2: reach the required blocks. [can be scheduled]
                    #Case 2.3: no return during adjusting both prefill and decode candidates. (not 
                    #reaching the required #blocks after checking all the candidates)
                    #[cannot be scheduled]
                    
                    num_accum_shared_blocks = 0
                    num_accum_blocks = 0
                    num_required_blocks = math.ceil(num_required_blocks/2)
                    num_invalid_prefills = 0 
                    
                    #Step 1: check previous prefill requests.
                    for i in range(len(seq_groups)):
                        seq = seq_groups[i].seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
                        if seq.use_hidden or seq.n_blocks == 1: #n_blocks =1 cannot be freed.
                            num_invalid_prefills += 1
                            continue
                        else:
                            #check whether there is enough shared blocks first.
                            if actual_num_free_shared_gpu_blocks >= \
                            seq.n_blocks + num_accum_shared_blocks: 
                                prefill_adjust.append(i)
                                num_accum_blocks += math.floor(seq.n_blocks/2)
                                num_accum_shared_blocks += seq.n_blocks
                                if num_accum_blocks >= num_required_blocks: 
                                    #shared blocks and private blocks are both not violated
                                    return AllocStatus.OK_HIDDEN, prefill_adjust, decode_adjust
                            else: #run out of shared blocks before reaching enough private blocks.
                                return AllocStatus.LATER, prefill_adjust, decode_adjust
                                
                    #Step 2: check whether modify decode candidates would help.
                    if num_invalid_prefills == len(seq_groups):
                        assert num_accum_blocks < num_required_blocks
                        assert actual_num_free_shared_gpu_blocks >= num_accum_shared_blocks
                        for i in range(len(backup_groups)):
                            seq = backup_groups[i].get_seqs(status=SequenceStatus.RUNNING)[0]
                            if seq.use_hidden or seq.n_blocks == 1:
                                continue
                            else:
                                num_tokens_temp = seq.get_len()
                                #schedule budget check first.
                                if num_curr_batched_tokens + num_tokens_temp > token_budget or \
                                num_curr_seqs + 1 > max_num_seqs:
                                    return AllocStatus.LATER, prefill_adjust, decode_adjust
                                else:
                                    #the reason for seq.n_blocks here is that when we schedule a 
                                    #transformation of a decode request, we actually still generate a new
                                    #token in a prefill/recompute way. So we may need to append the slot.
                                    if actual_num_free_shared_gpu_blocks >= \
                                    num_accum_shared_blocks + seq.n_blocks:
                                        num_curr_batched_tokens += num_tokens_temp
                                        num_curr_seqs += 1
                                        decode_adjust.append(i)
                                        num_accum_blocks += math.floor(seq.n_blocks/2)
                                        num_accum_shared_blocks += seq.n_blocks
                                        if num_accum_blocks >= num_required_blocks: 
                                            #shared blocks and private blocks are both not violated
                                            return AllocStatus.OK_HIDDEN, prefill_adjust, decode_adjust
                                    else:
                                        #run out of shared blocks before reaching enough private blocks.
                                        return AllocStatus.LATER, prefill_adjust, decode_adjust
            
                    assert num_accum_blocks < num_required_blocks
                    if num_accum_blocks < num_required_blocks:
                        #iterate over all the candidates, but cannot alloc due to not reaching
                        return AllocStatus.LATER, prefill_adjust, decode_adjust 

    def check_abnormal_request(self, seq_group: SequenceGroup) -> AllocStatus:
        self_num_required_blocks = self._get_seq_num_required_blocks(
            seq_group.get_seqs(status=SequenceStatus.WAITING)[0])
        num_required_blocks = self_num_required_blocks
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        else:
            return AllocStatus.OK
    
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        self_num_required_blocks = self._get_seq_num_required_blocks(
            seq_group.get_seqs(status=SequenceStatus.WAITING)[0])
        cross_num_required_blocks = self._get_seq_num_required_blocks(
            seq_group.get_encoder_seq())
        num_required_blocks = self_num_required_blocks + \
                              cross_num_required_blocks

        if self.block_sliding_window is not None:

            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER
    
    def _allocate_sequence_shared(self, \
                                  seq:Sequence, \
                                 ref_count: int) -> BlockTable:
        #we do not support encoder_decoder structure for now.
        num_prompt_blocks = seq.n_blocks
        block_table: BlockTable = []
        
        for logical_idx in range(num_prompt_blocks):
            block = self.gpu_allocator.allocate_shared()
            block.ref_count = ref_count
            block_table.append(block)
        return block_table
    
    def _allocate_sequence(self, \
                           seq: Sequence, \
                           ref_count: int, \
                           is_encoder_decoder: bool = True) -> BlockTable:
        # Allocate new physical token blocks that will store the prompt tokens.
#         if not seq.use_hidden:
#             num_prompt_blocks = len(seq.logical_token_blocks) #need to check this. (#CAUTION.)
#         else: #use_hidden requires only half of the blocks.
#             num_prompt_blocks = math.ceil(len(seq.logical_token_blocks)/2)
        
        #We should think about how to assign block_table for the hidden-cache-utilized requests.
        num_prompt_blocks = seq.n_blocks
        block_table: BlockTable = []
        for logical_idx in range(num_prompt_blocks):
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
                # Set the reference counts of the token blocks.
                block.ref_count = ref_count
            elif not is_encoder_decoder and self.enable_caching:
                block = self.gpu_allocator.allocate(
                    seq.hash_of_block(logical_idx),
                    seq.num_hashed_tokens_of_block(logical_idx))
            else:
                if not seq.use_hidden:
                    block = self.gpu_allocator.allocate()
                    # Set the reference counts of the token blocks.
                    block.ref_count = ref_count
                else: #this sequence is with hidden cache enabled.
                    if logical_idx % 2 == 0: #should assign a new cache, and take up the k cache.
                        block = self.gpu_allocator.allocate()
                        block.ref_count = ref_count
                    else: #in the same physical block but v cache.
                        block = cp.deepcopy(block) #inherit from the last created block.
                        block.block_number += self.num_total_gpu_blocks
                    
            block_table.append(block)
        
        return block_table
    
    #our modifications.
    def adjust_cache_block(self,
                           seq_group: SequenceGroup,
                           shared_only: bool = False,
                           reset_flag: bool = False) -> None:
        
        seq = seq_group.get_seqs()[0]
        seq_group.set_use_hidden()
        if not shared_only:
            block_table = self.block_tables[seq.seq_id]
            self.block_tables[seq.seq_id] = self._free_block_table_partial(block_table, seq)
        
        self.allocate_shared(seq_group)
        if reset_flag: #should be in-place operation.
            seq_group.seqs_dict[seq.seq_id].reset_state_for_recompute() #decode sets to prefill.
        
    
    def allocate_shared(self, seq_group:SequenceGroup) -> None:
        seq = seq_group.get_seqs()[0]
        block_table: BlockTable = \
            self._allocate_sequence_shared(seq, seq_group.num_seqs())
        for seq in seq_group.get_seqs():
            self.shared_block_tables[seq.seq_id] = block_table.copy()
    
    def allocate(self, seq_group: SequenceGroup) -> None:
        is_encoder_decoder = seq_group.is_encoder_decoder()
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        block_table: BlockTable = \
            self._allocate_sequence(seq,
                                    seq_group.num_seqs(),
                                    is_encoder_decoder)
        
        # Assign the self-attention block tables for each sequence.
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            self.block_tables[seq.seq_id] = block_table.copy()

        # Allocate encoder sequence
        if is_encoder_decoder:
            # A SequenceGroup has only a single encoder sequence (at most),
            # thus allocate with a ref count of 1
            block_table = self._allocate_sequence(seq_group.get_encoder_seq(),
                                                  1, is_encoder_decoder)
            # Assign the cross-attention block table for the SequenceGroup.
            self.cross_block_tables[seq_group.request_id] = block_table

    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0,
                         num_potential_blocks: int = 0,
                         num_potential_shared_blocks: int = 0) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerV1"

        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        shared_ok_flag = True
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks() - num_potential_blocks
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        if seq_group.use_hidden:
            num_free_shared_gpu_blocks = \
            self.gpu_allocator.get_num_free_shared_blocks() - num_potential_shared_blocks
            shared_ok_flag = num_free_shared_gpu_blocks > 1
        return (num_seqs <= num_free_gpu_blocks) & shared_ok_flag

    def _promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        assert self.enable_caching

        # Compute a new hash for the block so that it can be shared by other
        # Sequences
        new_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)

        # if new_hash is already in the cached table, then free last_block
        # and return the cached version
        if self.gpu_allocator.contains_block(new_hash):
            self.gpu_allocator.free(last_block)
            return self.gpu_allocator.allocate(new_hash)
        else:
            self.gpu_allocator.update_hash(new_hash, last_block)
            return last_block

    def _is_last_block_full(
        self,
        seq: Sequence,
    ) -> bool:
        token_ids_len = seq.data.get_len()
        return token_ids_len > 0 and token_ids_len % seq.block_size == 0

    def _maybe_promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        if self._is_last_block_full(seq):
            return self._promote_last_block(seq, last_block)
        else:
            return last_block
    
    def _allocate_last_shared_physical_block(
        self,
    ) -> PhysicalTokenBlock:
        return self.gpu_allocator.allocate_shared()
    
    def _allocate_last_physical_block(
        self,
        seq: Sequence,
    ) -> PhysicalTokenBlock:
        # Called before a new block is appended.
        # This is in charge of allocating a new physical block (to be appended).

        # None if the last block is not full. Otherwise, we set it to the
        # content hash.
        if not self.enable_caching:
            return self.gpu_allocator.allocate()
        block_hash: Optional[int] = None
        if (self._is_last_block_full(seq)):
            block_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)
        num_hashed_tokens = seq.num_hashed_tokens_of_block(
            len(seq.logical_token_blocks) - 1)

        # num_hashed_tokens is used to compute future hashes
        # (e.g. in the hashing function, it is used to ask the sequence for
        # prefix tokens)
        new_block = self.gpu_allocator.allocate(block_hash, num_hashed_tokens)

        # If the block has is None, then the block is not full.
        # If the block is not full, then we expect it to have a refcount of 1.
        if block_hash is None:
            assert new_block.ref_count == 1
        return new_block
    
    def check_increased_blocks(self,
                               seq: Sequence,) -> bool:
        #this function is used in 'schedule_running.' Need to distinguish whether the sequence
        #is in 'use_hidden' mode or not.
        #For 'use_hidden' requests, the actual number of existing used number of blocks and the 
        #acutal number of required blocks are as follows:
        #actual_len_block_table = len(block_table)//2 + len(block_table)%2
        #acutal_n_blocks = n_blocks // 2 + n_blocks % 2
        #Case I: say that len(block_table) = 7, n_blocks = 8.
        #then: actual_len_block_table = 4, actual_n_blocks = 4
        #we don't need to add a new physical block in private cache.
        #Case II: say that len(block_table) = 6, n_blocks = 7.
        #then: ctual_len_block_table = 3, actual_n_blocks = 4
        #then we need to add new physical block in private cache.
        n_blocks = seq.n_blocks
        block_table = self.block_tables[seq.seq_id]
        add_block_flag, add_shared_block_flag = False, False
        if not seq.use_hidden: #normal sequence using KV cache.
            if len(block_table) < n_blocks:
                assert len(block_table) == n_blocks - 1
                add_block_flag = True
        else: #sequence using hidden cache.
            actual_len_block_table = len(block_table) // 2 + len(block_table) % 2
            acutal_n_blocks = n_blocks // 2 + n_blocks % 2
            if len(block_table) < n_blocks: 
                assert len(block_table) == n_blocks - 1
                add_shared_block_flag = True
            if actual_len_block_table < acutal_n_blocks:
                assert actual_len_block_table == acutal_n_blocks - 1
                add_block_flag = True
                
        return add_block_flag, add_shared_block_flag
    
    def append_slots(
            self,
            seq: Sequence,
            num_lookahead_slots: int = 0,
        ) -> List[Tuple[int, int]]:
            """Allocate a physical slot for a new token."""
            n_blocks = seq.n_blocks
            block_table = self.block_tables[seq.seq_id]
            # If we need to allocate a new physical block
            if len(block_table) < n_blocks:
                # Currently this code only supports adding one physical block
                assert len(block_table) == n_blocks - 1

                if (self.block_sliding_window
                        and len(block_table) >= self.block_sliding_window):
                    # reuse a block
                    block_table.append(block_table[len(block_table) %
                                                   self.block_sliding_window])
                else:
                    #we should distinguish the requests with different cache type here.                   
                    if seq.use_hidden == False:
                        # Allocate a new physical block.
                        new_block = self._allocate_last_physical_block(seq)
                        block_table.append(new_block)
                    else:
                        if len(block_table) % 2 == 0:
                            new_block = self._allocate_last_physical_block(seq)
                        else:
                            new_block = cp.deepcopy(block_table[-1])
                            new_block.block_number += self.num_total_gpu_blocks
                        block_table.append(new_block)
                        block_table_4_shared = self.shared_block_tables[seq.seq_id]
                        new_shared_block = self._allocate_last_shared_physical_block()
                        block_table_4_shared.append(new_shared_block)
                        #allocate for shared blocks remember.
                        
                    #we should distinguish the requests with different cache type here.
                    
                    return []

            # We want to append the token to the last physical block.
            last_block = block_table[-1]
            assert last_block.device == Device.GPU
            if last_block.ref_count == 1:
                # Not shared with other sequences. Appendable.
                if self.enable_caching:
                    # If the last block is now complete, we may reuse an old block
                    # to save memory.
                    maybe_new_block = self._maybe_promote_last_block(
                        seq, last_block)
                    block_table[-1] = maybe_new_block
                return []
            else:
                # The last block is shared with other sequences.
                # Copy on Write: Allocate a new block and copy the tokens.
                new_block = self._allocate_last_physical_block(seq)

                block_table[-1] = new_block
                self.gpu_allocator.free(last_block)
                return [(last_block.block_number, new_block.block_number)]

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        # When using a sliding window, blocks will be eventually reused.
        # In this case the block tables will contain repeated blocks.
        # When forking, we must make sure that each block's `ref_count`
        # is only incremented by one, so we deduplicate them by wrapping
        # them in a set.
        for block in set(src_block_table):
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:

        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        request_id = seq_group.request_id
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        # Cross-attention blocks
        if seq_group.is_encoder_decoder():
            blocks.update(self.cross_block_tables[request_id])
        return list(blocks)

    def can_swap_in(self,
                    seq_group: SequenceGroup,
                    num_lookahead_slots: int = 0) -> AllocStatus:
        assert (num_lookahead_slots == 0
                ), "BlockSpaceManagerV1 does not support lookahead allocation"

        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        if seq_group.is_encoder_decoder():
            num_swapped_seqs += 1
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        if self.gpu_allocator.get_num_total_blocks() < num_required_blocks:
            return AllocStatus.NEVER
        elif num_free_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _swap_block_table(
            self, block_table: BlockTable, src_allocator: BlockAllocatorBase,
            dest_allocator: BlockAllocatorBase,
            mapping: Dict[PhysicalTokenBlock,
                          PhysicalTokenBlock]) -> BlockTable:
        new_block_table = []

        for from_block in block_table:
            if from_block in mapping:
                to_block = mapping[from_block]
                to_block.ref_count += 1
            else:
                to_block = dest_allocator.allocate(
                    from_block.block_hash, from_block.num_hashed_tokens)
                mapping[from_block] = to_block
            new_block_table.append(to_block)
            # Free the source block swapped in to destination.
            src_allocator.free(from_block)

        return new_block_table

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:

        request_id = seq_group.request_id

        # CPU block -> GPU block.
        # dict is efficient in lookup `if cpu_block in mapping`
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            self.block_tables[seq.seq_id] = \
                self._swap_block_table(self.block_tables[seq.seq_id],
                                       self.cpu_allocator,
                                       self.gpu_allocator,
                                       mapping)

        if seq_group.is_encoder_decoder():
            self.cross_block_tables[request_id] = \
                self._swap_block_table(self.cross_block_tables[request_id],
                                       self.cpu_allocator,
                                       self.gpu_allocator,
                                       mapping)

        return [(cpu_block.block_number, gpu_block.block_number)
                for cpu_block, gpu_block in mapping.items()]

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        request_id = seq_group.request_id

        # GPU block -> CPU block.
        # dict is efficient in lookup `if gpu_block in mapping`
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            self.block_tables[seq.seq_id] = \
                self._swap_block_table(self.block_tables[seq.seq_id],
                                       self.gpu_allocator,
                                       self.cpu_allocator,
                                       mapping)

        if seq_group.is_encoder_decoder():
            self.cross_block_tables[request_id] = \
                self._swap_block_table(self.cross_block_tables[request_id],
                                       self.gpu_allocator,
                                       self.cpu_allocator,
                                       mapping)

        return [(cpu_block.block_number, gpu_block.block_number)
                for cpu_block, gpu_block in mapping.items()]
                
    def _free_block_table_partial(self, block_table, seq):
        assert len(block_table) != 1 #no len(block_table) == 1 occurs.
        for i in range(len(block_table)):
            if i % 2 == 0:
                continue
            else:
                self.gpu_allocator.free(block_table[i])
                block_table[i] = cp.deepcopy(block_table[i-1])
                block_table[i].block_number += self.num_total_gpu_blocks
        return block_table
    
    
    def _free_shared_block_table(self, block_table: BlockTable) -> None:
        #not made sliding window + cpu_allocator compatible yet.
        blocks_to_free = block_table
        for block in set(blocks_to_free):
            self.gpu_allocator.free_shared(block)
    
    def _free_block_table_hidden(self, block_table: BlockTable) -> None:
        blocks_to_free = block_table[::2]
        for block in set(blocks_to_free):
            assert block.device == Device.GPU
            self.gpu_allocator.free(block)
        
    
    def _free_block_table(self, block_table: BlockTable) -> None:
        # when using a sliding window, each seq will only use up
        # to `self.block_sliding_window` blocks. When freeing
        # the block table, we must make sure to not free blocks more
        # than once. If no sliding window is used, there is no block
        # reuse in the block table, so we must free all blocks.
        blocks_to_free = (block_table[-self.block_sliding_window:]
                          if self.block_sliding_window is not None else
                          block_table)
        for block in set(blocks_to_free):
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)
    
    def free_shared(self, seq: Sequence) -> None:
        if seq.seq_id not in self.shared_block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.shared_block_tables[seq.seq_id]
        self._free_shared_block_table(block_table)
        del self.shared_block_tables[seq.seq_id]
    
    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        if not seq.use_hidden:
            self._free_block_table(block_table)
        else:
            self._free_block_table_hidden(block_table)
        del self.block_tables[seq.seq_id]

    def free_cross(self, seq_group: SequenceGroup) -> None:
        if seq_group.request_id not in self.cross_block_tables:
            # Already freed or hasn't ben scheduled yet.
            return
        block_table = self.cross_block_tables[seq_group.request_id]
        self._free_block_table(block_table)
        del self.cross_block_tables[seq_group.request_id]

    def reset(self) -> None:
        # Free decoder block tables
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()
        # Free cross-attention block tables
        for block_table in self.cross_block_tables.values():
            self._free_block_table(block_table)
        self.cross_block_tables.clear()
    
    def get_shared_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.shared_block_tables[seq.seq_id]
        return [block.block_number for block in block_table]
    
    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        block_table = self.cross_block_tables[seq_group.request_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        if self.enable_caching:
            # Update the last accessed time of all the blocks accessed
            # in this step.
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                block.last_accessed = access_time

    def compute_full_blocks_in_seq(self, seq: Sequence):
        if seq.seq_id not in self.block_tables:
            return
        max_full_block = seq.get_len() // self.block_size - 1
        block_table = self.block_tables[seq.seq_id]
        if max_full_block == -1:
            return
        for i in reversed(range(max_full_block)):
            if block_table[i].computed:
                break
            block_table[i].computed = True

    def get_all_computed_blocks(self, seq: Sequence) -> List[int]:
        if seq.seq_id not in self.block_tables:
            return []
        block_table = self.block_tables[seq.seq_id]
        # NOTE We exclude the last block to avoid the case where the entire
        # prompt is cached. This would cause erroneous behavior in model
        # runner.
        return [
            b.block_number
            for b in takewhile(lambda b: b.computed, block_table[:-1])
        ]

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Return the block ids that are common for a given sequence group.

        Used in prefill (can skip prefill of some blocks).
        """
        # Can return non-empty result only with prefix caching enabled.
        if not self.enable_caching:
            return []

        ids_list = [self.get_all_computed_blocks(seq) for seq in seqs]
        return commonprefix([ids for ids in ids_list if ids != []])

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        if self.enable_caching:
            for seq in seq_group.seqs_dict.values():
                self.compute_full_blocks_in_seq(seq)
