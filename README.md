# NxFixed

Fixed-point arithmetic for Nx with proven ranges.

- **Range analysis:** `NxFixed.analyze/3` traces a `defn`, builds a
  dataflow graph and computes the set of values every node can take
  (`NxFixed.IntSet`: unions of intervals, with case splits over
  small-domain inputs). The width check refuses kernels where a value can
  leave its type and says where and by how much.
- **Number formats:** `NxFixed.Num` (`float()` or `q(bits, frac,
  rounding: ...)`) lets one algorithm run as a float reference or in fixed
  point; `NxFixed.Primitives` has the rounding shifts, saturation and
  lookup-table NCO those formats need.
- **Checked execution:** `NxFixed.run_scan/5` and `run_scan_stacked/5` run
  stream kernels `step(sample, state, params) -> {output, next_state}` on
  the CPU, width-checked once, in one compiled loop with EXLA.

Used by [nx_simd](../nx_simd) (native code for integer graphs),
[ex_sdr](../ex_sdr) (radio building blocks) and [ex_hdl](../exhdl)
(hardware generation).

```elixir
NxFixed.analyze(&MyDSP.mixer/3, [Nx.template({}, :s16), Nx.template({}, :s16), Nx.template({}, :u16)])
#=> {:ok, graph} | {:error, %NxFixed.RangeError{message: "multiply needs -1073709056..1073741824 ..."}}
```
