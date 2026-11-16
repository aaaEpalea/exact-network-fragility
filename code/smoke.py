import numpy as np
from tracr.rasp import rasp
from tracr.compiler import compiling

def make_length():
    all_true = rasp.Select(rasp.tokens, rasp.tokens, lambda k, q: True)
    return rasp.SelectorWidth(all_true)

def make_reverse():
    opp = rasp.Map(lambda length: length - 1, make_length())
    opp = rasp.SequenceMap(lambda length, i: length - i - 1, make_length(), rasp.indices)
    sel = rasp.Select(rasp.indices, opp, rasp.Comparison.EQ)
    return rasp.Aggregate(sel, rasp.tokens)

prog = make_reverse()
model = compiling.compile_rasp_to_model(prog, vocab={1,2,3,4,5}, max_seq_len=6, compiler_bos="BOS")
out = model.apply(["BOS",1,2,3,4])
print(out.decoded, out.transformer_output.shape if hasattr(out,'transformer_output') else None)
print({k: v.shape for k,v in model.params.items()} if hasattr(model,'params') else 'no params')
