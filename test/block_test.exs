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

  test "frame blocks declare their rate" do
    b = %Block{name: :pairs, kind: :frame, fun: &pair_sums/3, rate: {2, 1}}
    {y, _} = Block.run(b, Nx.tensor([1, 2, 3, 4]), Nx.tensor(0))
    assert Nx.to_list(y) == [3, 7]
    assert Block.output_items(b, 4) == 2
  end
end
