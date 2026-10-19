defmodule NxFixed.Block do
  @moduledoc """
  What a signal processing block declares about itself, as plain data, so
  that the CPU runtime (chunking, threads) and hardware generators can plan
  for it without knowing what the block does.

    * `:kind`
      * `:stream` - `step(sample, state, params..., opts) -> {output, next_state}`
        once per item (see `NxFixed.run_scan/5`).
      * `:frame` - `process(items, state, params..., opts) -> {outputs, next_state}`
        on `rate` sized groups of items, `items` with the item index as
        first axis (FFT, interleaver blocks, codes).
      * `:host` - plain Elixir on binaries or lists (control, framing);
        not meant for hardware dataflow, `fun.(input, state)`.
    * `:fun` - the function above; `:params` - its constant tensor
      arguments; `:opts` - its trailing keyword options.
    * `:init` - the initial state; `nil` for stateless blocks, whose
      stream kernels then get a `u8` zero as state and return it.
    * `:rate` - `{in, out}`: every `in` input items give `out` output
      items (`{1, 1}` sample by sample, `{4, 1}` decimation by 4, `{188, 204}`
      an RS(204, 188) encoder on bytes).
    * `:latency` - output items of delay before the first item that depends
      on the input (filter group delay, interleaver delay); `:warmup` -
      input items after which the output no longer depends on `:init`
      (needed to start a copy of the block mid-stream, e.g. Viterbi chunks
      decoded in parallel), or `:unbounded` when the state can remember
      the start indefinitely (hysteresis, a PLL without lock guarantee).
    * `:item` - `{in_type, out_type}`, e.g. `{:complex, :complex}`,
      `{:bit, :symbol}`, `{:byte, :byte}`, `{:real, :real}`.
    * `:hardware` - which hardware structure it needs:
      * `:stateless` - pure function per item
      * `:feed_forward` - state is a delay line only (FIR, differential
        encoder's input side); pipelines freely
      * `:feedback` - state depends on itself (IIR, NCO, PLL, Costas,
        scramblers); minimum initiation interval from the loop
      * `:memory` - large state addressed by a counter (interleavers,
        Viterbi traceback): block RAM
      * `:transform` - whole-frame transform (FFT, LDPC): a dedicated core
      * `:control` - data dependent control flow (sync search, framing):
        state machine or the CPU
    * `:gnuradio` - the GNU Radio block it is checked against, as import
      path (`"filter.fir_filter_ccf"`), or `nil`.

  A module providing blocks implements `c:block/2`.
  """

  @enforce_keys [:name, :kind, :fun]
  defstruct [
    :name,
    :kind,
    :fun,
    :init,
    :gnuradio,
    params: [],
    opts: [],
    rate: {1, 1},
    latency: 0,
    warmup: 0,
    item: {:real, :real},
    hardware: :stateless
  ]

  @type kind :: :stream | :frame | :host
  @type hardware :: :stateless | :feed_forward | :feedback | :memory | :transform | :control
  @type t :: %__MODULE__{
          name: atom,
          kind: kind,
          fun: function,
          init: term,
          gnuradio: String.t() | nil,
          params: list,
          opts: keyword,
          rate: {pos_integer, pos_integer},
          latency: non_neg_integer,
          warmup: non_neg_integer | :unbounded,
          item: {atom, atom},
          hardware: hardware
        }

  @doc "The block `name` configured by `opts` (including `:format` where it applies)."
  @callback block(name :: atom, opts :: keyword) :: t

  @doc """
  Runs `block` over `input` from `state` (default: `block.init`); returns
  `{output, next_state}`. For `:stream` blocks `input` has the item index
  as first axis (`NxFixed.run_scan_stacked/5`, options passed on); for
  `:frame` blocks its length must be a multiple of the input rate.
  """
  def run(block, input, state \\ :init, opts \\ [])

  def run(%__MODULE__{} = b, input, :init, opts), do: run(b, input, b.init, opts)

  # Stateless stream kernels still take and return a state: a u8 zero.
  def run(%__MODULE__{kind: :stream} = b, input, nil, opts) do
    {output, _} = run(b, input, Nx.tensor(0, type: :u8), opts)
    {output, nil}
  end

  def run(%__MODULE__{kind: :stream} = b, input, state, opts),
    do: NxFixed.run_scan_stacked(b.fun, input, state, b.params, [kernel_opts: b.opts] ++ opts)

  def run(%__MODULE__{kind: :frame} = b, input, state, _opts),
    do: apply(b.fun, [input, state | b.params] ++ [b.opts])

  def run(%__MODULE__{kind: :host} = b, input, state, _opts), do: b.fun.(input, state)

  @doc """
  Output items for `n` input items (`n` a multiple of the input rate).
  """
  def output_items(%__MODULE__{rate: {i, o}}, n) when rem(n, i) == 0, do: div(n, i) * o
end
