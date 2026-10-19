defmodule NxFixed.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_fixed,
      version: "0.1.0",
      elixir: "~> 1.17",
      description:
        "Fixed-point arithmetic for Nx with proven ranges (interval range analysis, width check, number formats)",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 1.0"},
      {:exla, "~> 1.0", optional: true},
      {:stream_data, "~> 1.1", only: [:dev, :test]}
    ]
  end
end
