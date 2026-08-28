import torch

from elda.datamodules.data.circuit_pin_slot_tokenizer import PinSlotSpec
from elda.datamodules.data.circuit_pin_slot_v2_1_tokenizer import CircuitPinSlotV21GCTokenizer
from elda.models.seq_models import PinSlotGrammar


def _tokenizer():
    specs = {
        "NAND2_X1": PinSlotSpec(["A1", "A2"], ["ZN"], 1),
        "FA_X1": PinSlotSpec(["A", "B", "CI"], ["S", "CO"], 2),
        "HA_X1": PinSlotSpec(["A", "B"], ["S", "CO"], 2),
        "DFF_X1": PinSlotSpec(["D", "CK"], ["Q", "QN"], 1),
        "DFFR_X1": PinSlotSpec(["D", "CK", "RN"], ["Q", "QN"], 1),
        "MUX2_X1": PinSlotSpec(["A", "B", "S"], ["Z"], 1),
        "OAI21_X1": PinSlotSpec(["A", "B1", "B2"], ["ZN"], 1),
        "AOI21_X1": PinSlotSpec(["A", "B1", "B2"], ["ZN"], 1),
    }
    label_to_cell = {i: name for i, name in enumerate(specs)}
    pin_names = sorted({p for spec in specs.values() for p in [*spec.inputs, *spec.outputs]})
    tok = CircuitPinSlotV21GCTokenizer(
        label_to_cell=label_to_cell,
        pin_specs=specs,
        net_id=999,
        boundary_stub_id=1000,
        gate_count_buckets=[64, 80, 96, 112, 128, 160, 224, 320],
    )
    tok.set_num_nodes(64)
    tok.set_pin_name_vocab(pin_names)
    tok.set_num_node_and_edge_types(num_node_types=max(label_to_cell) + 1)
    tok.max_gates = 16
    tok.max_extra_pins_per_gate = 0
    return tok


def _allowed_after(tok, seq):
    proc = PinSlotGrammar(tok, 1, "cpu", mask_mode="practical_generation")
    scores = None
    for token in seq:
        scores = torch.zeros((1, len(tok)), dtype=torch.float32)
        scores = proc(torch.tensor([[int(token)]], dtype=torch.long), scores)
    return set(int(i) for i in torch.isfinite(scores[0]).nonzero(as_tuple=False).flatten().tolist())


def test_v2_1_required_and_output_pin_names_are_slot_exact():
    tok = _tokenizer()
    for label, cell_name in tok.label_to_cell.items():
        spec = tok.pin_specs[cell_name]
        seq = [tok.sos, tok.gate, tok._tok_endpoint_idx(tok.ep_net, 0), tok._tok_cell_label(label)]
        plan = [("PIN_IN", p) for p in spec.inputs]
        min_out = max(1 if spec.outputs else 0, int(spec.min_required_outputs))
        plan += [("PIN_OUT", p) for p in spec.outputs[:min_out]]
        plan += [("PIN_OPT_OUT", p) for p in spec.outputs[min_out:]]
        for role_name, pin_name in plan:
            role_token = getattr(tok, role_name.lower())
            assert role_token in _allowed_after(tok, seq)
            seq.append(role_token)
            assert _allowed_after(tok, seq) == {tok._tok_pin(pin_name)}
            seq.append(tok._tok_pin(pin_name))
            if role_name == "PIN_IN":
                seq.extend([tok.ep_load_stub, tok._tok_endpoint_idx(tok.ep_load_stub, len(seq))])
            elif role_name == "PIN_OUT":
                seq.extend([tok.ep_driver_stub, tok._tok_endpoint_idx(tok.ep_driver_stub, len(seq))])
            else:
                seq.extend([tok.ep_optional_driver_stub, tok._tok_endpoint_idx(tok.ep_optional_driver_stub, len(seq))])


def test_v2_1_pin_skip_only_for_optional_outputs():
    tok = _tokenizer()
    label = next(k for k, v in tok.label_to_cell.items() if v == "DFF_X1")
    seq = [tok.sos, tok.gate, tok._tok_endpoint_idx(tok.ep_net, 0), tok._tok_cell_label(label)]
    for pin in ["D", "CK", "Q"]:
        role = tok.pin_in if pin in {"D", "CK"} else tok.pin_out
        assert tok.pin_skip not in _allowed_after(tok, seq)
        seq.extend([role, tok._tok_pin(pin)])
        ep = tok.ep_load_stub if role == tok.pin_in else tok.ep_driver_stub
        seq.extend([ep, tok._tok_endpoint_idx(ep, len(seq))])
    allowed = _allowed_after(tok, seq)
    assert tok.pin_opt_out in allowed
    assert tok.pin_skip in allowed
