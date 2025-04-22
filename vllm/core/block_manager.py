"""A block manager that manages token blocks."""
import enum
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device


class BlockAllocator:
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
        # for i in range(num_blocks):
        for i in range(num_blocks-1, -1, -1):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)




    # <>
    # remove the parameter tmp_blk_idxs after we check the correctness
    def allocate_given_blk_idxs_in_free_list(self, blk_idxs: List[int]) -> PhysicalTokenBlock:
        # block = self.free_blocks.pop(blk_idx)
        # block.ref_count = ref_count
        # return block
        new_free_blocks: BlockTable = []
        blocks: BlockTable = []
        for blk_i, block in enumerate(self.free_blocks):
            if blk_i in blk_idxs:
                blocks.append(block)
                if block.ref_count == 0:
                    block.ref_count = 1
                    # assert blk_i in tmp_blk_idxs
            else:
                new_free_blocks.append(block)
        self.free_blocks = new_free_blocks
        return blocks






class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}


        # <>
        self.actual_gpu_blk_rng_end: int = self.num_total_gpu_blocks        

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = len(seq.logical_token_blocks)

        if seq_group.prefix is not None and seq_group.prefix.allocated:
            num_required_blocks -= seq_group.prefix.get_num_blocks()

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

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        num_prompt_blocks = len(seq.logical_token_blocks)

        block_table: BlockTable = []
        prefix_block_table: BlockTable = []
        num_prefix_blocks = 0

        prefix = seq_group.prefix
        if prefix is not None and prefix.allocated:
            # Prefix has already been allocated. Use the existing block table.
            num_prompt_blocks -= prefix.get_num_blocks()
            for block in prefix.block_table:
                block.ref_count += seq_group.num_seqs()
                block_table.append(block)

        for logical_idx in range(num_prompt_blocks):
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
            else:
                block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        if prefix is not None and not prefix.allocated:
            # Allocate blocks for the prefix, we will compute the prefix's
            # KV cache in this run.
            num_prefix_blocks = prefix.get_num_blocks()
            prefix_block_table = block_table[:num_prefix_blocks]
            for block in prefix_block_table:
                block.ref_count += 1
            prefix.set_block_table(prefix_block_table)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                # re-use a block
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            else:
                # The sequence has a new logical block.
                # Allocate a new physical block.
                block = self.gpu_allocator.allocate()
                block_table.append(block)
                return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        if seq_group.prefix is not None:
            # make sure to swap in the prefix first
            assert seq_group.prefix.allocated and seq_group.prefix.computed

        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]
            if seq_group.prefix is not None:
                for block in seq_group.prefix.block_table:
                    new_block_table.append(block)
                    block.ref_count += 1

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if (seq_group.prefix is not None
                        and gpu_block in seq_group.prefix.block_table):
                    # NOTE: We do not swap out the prefix blocks for now.
                    self.gpu_allocator.free(gpu_block)
                    continue

                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        # for block in set(block_table):
        # TODO (): why there is a set here in the original vllm code????
        for block in block_table:
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()




    # <>
    def reorganize_gpu_blocks(self, num_layer_to_load: int) -> Dict[int, int]:
        '''
            Reorganize allocated GPU blocks to get enough continuous block ranges.
            This is used when the remaining requests is not enough to make the computation time 
            cover the weight loading time.
            Input:
                num_layer_to_load: int, how many layers' weights we need to load
            Changed:
                block_table, free_block_list, block.ref_count
            Output:
                the mapping {from_blk.block_number : to_blk.block_number}
        '''
        # update KVBlkPerLayerWeight.cached_layer_num
        KVBlkPerLayerWeight.cached_layer_num = KVBlkPerLayerWeight.cached_layer_num - num_layer_to_load
        KVBlkPerLayerWeight.load_more_layer_on_card_num = num_layer_to_load

        num_blk_per_layer = KVBlkPerLayerWeight.blk_num_per_layer
        tot_blk_num = num_blk_per_layer * num_layer_to_load
        
        release_rng_start = self.actual_gpu_blk_rng_end - tot_blk_num
        release_rng_end = self.actual_gpu_blk_rng_end
        # update the new KV cache block range end
        self.actual_gpu_blk_rng_end = release_rng_start
        
        # get the blocks to remove, to move to, and to allocate
        to_remove: List[int] = []
        blk_is_free = [False] * self.num_total_gpu_blocks
        blks_to_move_to: List[Tuple[int, PhysicalTokenBlock]] = list()
        blk_to_allocate_index_in_freelist = []
        blks_to_move_to_the_same_pos_in_new_KVcache: List[Tuple[int, int]] = list()
        for i, blk in enumerate(self.gpu_allocator.free_blocks):
            blk_is_free[blk.block_number]=True
            if blk.block_number < release_rng_start:
                blks_to_move_to.append((i, blk))
            else:
                blk_to_allocate_index_in_freelist.append(i)
        for blk_i in range(release_rng_start):
            if blk_is_free[blk_i] == False:
                blks_to_move_to_the_same_pos_in_new_KVcache.append((blk_i, blk_i))
        for blk_i in range(release_rng_start, release_rng_end):
            if blk_is_free[blk_i] == False:
                to_remove.append(blk_i)
        
        # get the blocks to move to
        assert len(blks_to_move_to) >= len(to_remove)
        blks_to_move_to = blks_to_move_to[:len(to_remove)]

        # change the related block_table information
        # also update the from_blk.ref_count to 1
        remove_mapping: Dict[int, Tuple[PhysicalTokenBlock, int]] = {from_blk_number: to_blk for from_blk_number, (_, to_blk) in zip(to_remove, blks_to_move_to)}
        for seq_i in self.block_tables:
            for blk_i, from_blk in enumerate(self.block_tables[seq_i]):
                ori_blk_number = from_blk.block_number
                if ori_blk_number in remove_mapping:
                    to_blk = remove_mapping[ori_blk_number]
                    to_blk.ref_count = from_blk.ref_count
                    from_blk.ref_count = 1
                    self.block_tables[seq_i][blk_i] = to_blk
        
        # allocate blks we need
        # for blk_i in blk_to_allocate_index_in_freelist:
        #     self.gpu_allocator.allocate_given_blk_idx_in_free_list(blk_i, 1)
        # for blk_i, blk in blks_to_move_to:
        #     self.gpu_allocator.allocate_given_blk_idx_in_free_list(blk_i, blk.ref_count)
        self.gpu_allocator.allocate_given_blk_idxs_in_free_list(blk_to_allocate_index_in_freelist + [blk_i for blk_i, _ in blks_to_move_to])

        # return the block number mapping information, so that we can do memory transfer
        ret: Dict[int, int] = {from_blk_number: to_blk.block_number for from_blk_number, (_, to_blk) in zip(to_remove, blks_to_move_to)}
        ret.update(blks_to_move_to_the_same_pos_in_new_KVcache)
        return ret

        





    # <>
    def get_dependent_blk_moving_chains(self, fromblknum_to_blknum: Dict[int, int]) -> Tuple[List[int], List[int]]:
        '''
            NOTE: this function is called after reorganize_gpu_blocks.
            Get the blk moving chains where there is dependency.
            E.g., (1) move blk 3 to blk 1, (2) blk 5 to blk 3. then we cannot do (1)&(2) together, and (1) and (2) form a chain.
            Input:
                fromblknum_to_blknum: Dict[int, int]. 
            Output:
                (the chains connected together, the length of each chain): Tuple[List[Tuple[int, int]], List[int]].
        '''
        print(f"KVBlkPerLayerWeight.layer_num: {KVBlkPerLayerWeight.layer_num}")
        layer_num = KVBlkPerLayerWeight.layer_num
        
        curr_tot_gpu_blk_num = self.actual_gpu_blk_rng_end
        ori_tot_gpu_blk_num = curr_tot_gpu_blk_num + \
            KVBlkPerLayerWeight.load_more_layer_on_card_num * KVBlkPerLayerWeight.blk_num_per_layer

        print(f"curr_tot_gpu_blk_num: {curr_tot_gpu_blk_num}, ori_tot_gpu_blk_num: {ori_tot_gpu_blk_num}, load_more_layer_on_card_num: {KVBlkPerLayerWeight.load_more_layer_on_card_num}, blk_num_per_layer: {KVBlkPerLayerWeight.blk_num_per_layer}")
        
        mapping_dict = fromblknum_to_blknum.copy()
        # get block mapping in the whole gpu cache
        for layer_i in range(1, 2*layer_num):
            # deal with key cache and value cache in every layer (except key cache in layer 0)
            mapping_dict.update([(k + ori_tot_gpu_blk_num * layer_i, v + curr_tot_gpu_blk_num * layer_i) \
                                 for k, v in fromblknum_to_blknum.items()])

        # print(f"mapping_dict: {mapping_dict}")
        
        # get dependent block mapping chains
        from_blk_gids = sorted(mapping_dict.keys())
        visited: Dict[int, int] = {gid: False for gid in from_blk_gids}
        chains: List[int] = list()
        chain_lens: List[int] = [0]
        for i in range(len(from_blk_gids)-1, -1, -1):
            src_gid = from_blk_gids[i]
            if visited[src_gid]:
                continue
            if mapping_dict[src_gid] == src_gid:
                visited[src_gid] = True
                continue
            chains.append(src_gid)
            while(src_gid in mapping_dict):
                visited[src_gid] = True
                src_gid = mapping_dict[src_gid]
                chains.append(src_gid)

            chain_lens.append(len(chains))

        # print(f"chains: {chains}")
        # print(f"chain_lens: {chain_lens}")

        return chains, chain_lens












class KVBlkPerLayerWeight:
    """
    Store the number of KV cache blocks to release if we want to store the weights of a layer in an LLM.
    blk_num_per_layer: int = -1
    block_size: int = -1 (in bytes)
    layer_weight_size: int = -1 (in bytes) 
    cached_layer_num: int = -1
    layer_num: int = -1 (the number of layers in the model)
    tot_gpu_mem: int = -1 (in bytes) total gpu memory of the card (regardless of the gpu utilization ratio)
    """
    blk_num_per_layer: int = -1
    block_size: int = -1
    layer_weight_size: int = -1
    cached_layer_num: int = -1
    load_more_layer_on_card_num: int = 0
    layer_num: int = -1
    tot_gpu_mem: int = -1

    @classmethod
    def reset(cls):
        cls.blk_num_per_layer: int = -1
        cls.block_size: int = -1
        cls.layer_weight_size: int = -1
        cls.cached_layer_num: int = -1
        cls.load_more_layer_on_card_num: int = 0
        cls.layer_num: int = -1    
        cls.tot_gpu_mem: int = -1

    @classmethod
    def print_info(cls):
        print(f"blk_num_per_layer={cls.blk_num_per_layer}, "
              f"block_size={cls.block_size}, "
              f"layer_weight_size={cls.layer_weight_size}, "
              f"cached_layer_num={cls.cached_layer_num}, "
              f"load_more_layer_on_card_num={cls.load_more_layer_on_card_num}, "
              f"layer_num={cls.layer_num}, "
              f"tot_gpu_mem={cls.tot_gpu_mem}",
              flush=True
              )