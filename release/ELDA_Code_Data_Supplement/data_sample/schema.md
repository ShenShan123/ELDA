# Frozen V6.1 Source--Demand schema

The sequence encodes four ordered sections after `SOS` and terminates with
`EOS`:

1. `CELL_SECTION`: `CELL_ID` pointer and `CELL_TYPE` for each standard cell.
2. `DEMAND_SECTION`: one object per Liberty input-pin slot, with a demand
   pointer, load-cell pointer, and load-pin token.
3. `SOURCE_SECTION`: source identity, source kind, optional source-cell
   pointer, source pin, optional boundary identity, source role, fanout
   bucket, and the `(max, used, remaining)` count triple.
4. `SOURCE_NET_SECTION`: one `SOURCE_LOAD` record per source.  Each record
   contains a source pointer, a count token, and exactly that many demand
   pointers.

Pointer and count tokens use disjoint contiguous ranges determined by
`max_num_nodes`. Pin and cell-type tokens follow those ranges. The executable
definitions are the frozen `CircuitSourceNetV61Tokenizer`, `serialize_v61`,
`decode_v61`, and `SourceNetV61Grammar` included under `src/`.

`SOURCE_KIND` values are `CELL_OUTPUT`, `BOUNDARY_SOURCE`, and
`CONSTANT_SOURCE`. Their corresponding `SOURCE_ROLE` values are `OUTPUT_PIN`,
`BOUNDARY_PIN`, and `CONST_PIN`. Every demand must be assigned exactly once;
every source has one source-load record; realized use must equal `used_fanout`;
and `used_fanout + remaining_fanout = max_fanout`.
