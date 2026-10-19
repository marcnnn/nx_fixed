defmodule NxFixed.WideFormatsTest do
  use ExUnit.Case, async: true
  import Nx.Defn
  import Bitwise

  alias NxFixed.{Num, Primitives}

  defn wrap32(a, b), do: Primitives.wrap(Nx.as_type(a, :s64) + b, :s32)
  defn round_shift(x), do: Primitives.shift_round(Nx.left_shift(Nx.as_type(x, :s64), 20), 33)

  # A 24-bit value in a 40-bit format: the product needs 47 bits.
  defn requantize(x, opts \\ []) do
    x = %NxFixed.Fixed{x | t: Nx.as_type(x.t, :s64)}
    Num.quantize(Num.mul(x, x), opts[:format])
  end

  test "wrap from s64 to s32" do
    assert {:ok, _} = NxFixed.analyze(&wrap32/2, [Nx.template({}, :s32), Nx.template({}, :s32)])
    one = Nx.tensor(1, type: :s32)
    assert Nx.to_number(wrap32(Nx.tensor(2_147_483_647, type: :s32), one)) == -2_147_483_648
  end

  test "rounding shifts of 32 bits and more" do
    assert {:ok, _} = NxFixed.analyze(&round_shift/1, [Nx.template({}, :s32)])
    # (3 * 2^12) << 20 = 3 * 2^32; rounded >> 33 gives 1.5 -> 2.
    assert Nx.to_number(round_shift(Nx.tensor(3 <<< 12, type: :s32))) == 2
  end

  test "formats wider than 32 bits" do
    f = Num.q(40, 30)
    x = %NxFixed.Fixed{t: Nx.tensor(round(0.75 * (1 <<< 22)), type: :s32), frac: 22}
    assert {:ok, _} = NxFixed.analyze(&requantize/2, [x, [format: f]])
    assert_in_delta Num.to_float(requantize(x, format: f)), 0.5625, 1.0e-9
    assert requantize(x, format: f).frac == 30
  end

  defn stacked(a, b), do: Nx.stack([Nx.as_type(a, :s32) + b, a])

  test "Nx.stack is traced as views" do
    assert {:ok, _} =
             NxFixed.analyze(&stacked/2, [Nx.template({2}, :s16), Nx.template({2}, :s16)])

    a = Nx.tensor([1, 2], type: :s16)
    assert Nx.to_list(stacked(a, a)) == [[2, 4], [1, 2]]
  end

  defn csum(a, b), do: Num.csub(Num.cadd(a, b), b)

  test "complex add and subtract" do
    f = Num.q(16, 15)
    a = {Num.constant(0.5, f), Num.constant(-0.25, f)}
    b = {Num.constant(0.125, f), Num.constant(0.5, f)}
    {re, im} = csum(a, b)
    assert {Num.to_float(re), Num.to_float(im)} == {0.5, -0.25}
  end
end
