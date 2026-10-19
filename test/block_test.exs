defmodule NxFixed.BlockTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  alias NxFixed.Block

  defn accumulate(x, acc, _opts \\ []) do
    y = NxFixed.Primitives.wrap(Nx.as_type(acc, :s32) + x, :s16)
    {y, y}
  end

  defn pair_sums(x, state, _opts \\ []), do: {Nx.sum(Nx.reshape(x, {:auto, 2}), axes: [1]), state}

  test "stream blocks thread their state, split anywhere" do
    b = %Block{
      name: :acc,
      kind: :stream,
      fun: &accumulate/3,
      init: Nx.tensor(0, type: :s16),
      hardware: :feedback
    }

    x = Nx.tensor([1, 2, 3, 4, 5], type: :s16)

    {all, final} = Block.run(b, x, :init, compiler: Nx.Defn.Evaluator)
    {a, s} = Block.run(b, x[0..1], :init, compiler: Nx.Defn.Evaluator)
    {c, ^final} = Block.run(b, x[2..4], s, compiler: Nx.Defn.Evaluator)

    assert Nx.to_list(all) == [1, 3, 6, 10, 15]
    assert Nx.to_list(Nx.concatenate([a, c])) == Nx.to_list(all)
  end

  defn double(x, state, _opts \\ []), do: {Nx.as_type(x, :s32) * 2, state}

  test "stateless stream blocks run without a state" do
    b = %Block{name: :double, kind: :stream, fun: &double/3}
    {y, nil} = Block.run(b, Nx.tensor([1, -2], type: :s16), :init, compiler: Nx.Defn.Evaluator)
    assert Nx.to_list(y) == [2, -4]
  end

  test "frame blocks declare their rate" do
    b = %Block{name: :pairs, kind: :frame, fun: &pair_sums/3, rate: {2, 1}}
    {y, _} = Block.run(b, Nx.tensor([1, 2, 3, 4]), Nx.tensor(0))
    assert Nx.to_list(y) == [3, 7]
    assert Block.output_items(b, 4) == 2
  end

  # Upsampling by 3/2 with a phase counter: 1 or 2 outputs per input.
  defn three_halves(x, phase, _opts \\ []) do
    count = Nx.select(phase == 1, 2, 1) |> Nx.as_type(:u8)
    {{Nx.stack([x, x]), count}, Nx.as_type(Nx.select(phase == 1, 0, 1), :u8)}
  end

  test "variable-rate blocks return only the valid items, split anywhere" do
    b = %Block{
      name: :up,
      kind: :stream,
      fun: &three_halves/3,
      init: Nx.tensor(0, type: :u8),
      rate: {:variable, 2}
    }

    x = Nx.tensor([1, 2, 3, 4], type: :s16)
    {y, s} = Block.run(b, x, :init, compiler: Nx.Defn.Evaluator)
    assert Nx.to_list(y) == [1, 2, 2, 3, 4, 4]
    {a, s1} = Block.run(b, x[0..0], :init, compiler: Nx.Defn.Evaluator)
    {c, ^s} = Block.run(b, x[1..3], s1, compiler: Nx.Defn.Evaluator)
    assert Nx.to_list(a) ++ Nx.to_list(c) == Nx.to_list(y)
    assert Block.compact(Nx.tensor([[1, 2]]), Nx.tensor([0])) == nil
  end
end
