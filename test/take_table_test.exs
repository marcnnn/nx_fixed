defmodule NxFixed.TakeTableTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  @table Nx.tensor([[1, 1000], [2, -2000], [3, 3]], type: :s16)

  # Row 1 only: its values are 2 and -2000, so the sum fits s16 only
  # if the rule reads rows, not flat elements 0..1.
  defn row_sum(i) do
    idx = Nx.clip(i, 1, 1)
    rows = Nx.take(@table, idx)
    Nx.sum(Nx.as_type(rows, :s32))
  end

  test "take on a 2-D constant table bounds by the selected rows" do
    {:ok, graph} = NxFixed.analyze(&row_sum/1, [Nx.template({}, :s32)])
    take = Enum.find(Map.values(graph.nodes), &(&1.op == :take))
    assert NxFixed.IntSet.to_list(take.range) |> Enum.min_max() == {-2000, 2}
  end
end
