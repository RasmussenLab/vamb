using CairoMakie
using DataFrames
using CSV
using Statistics
using CategoricalArrays

df = CSV.read("benchmarks.csv", DataFrame)

sort!(df, [:version, :dataset])

df.version = categorical(df.version)
df.dataset = categorical(df.dataset)

n_versions = length(levels(df.version))
n_datasets = length(levels(df.dataset))

summary = combine(
    groupby(df, [:version, :dataset]),
    :nc => mean => :nc_mean,
    :mq => mean => :mq_mean,
    :seconds => (i -> minimum(i) / 3600) => :hours_min,
)

combined = combine(
    groupby(summary, [:version]),
    :hours_min => sum => :hours_total
)

colors = Makie.wong_colors();
fig = Figure();

grid = fig[1, 1:2] = GridLayout()

# Plot accuracy
let
    global ax_accuracy = Axis(
        grid[1, 1],
        title = "Accuracy",
        xticks = (
            (((n_versions - 1) / 2) + 1):(n_versions + 1):(n_datasets * (n_versions + 1)),
            levels(summary.dataset),
        ),
        limits = (nothing, nothing, 0, nothing),
        ylabel = "Genomes recovered as NC / MQ bins",
        ylabelpadding = 0,
        xlabelpadding = -15,
        xticklabelrotation = 0.2,
        xticklabelpad = 10,
        xticklabelalign = (:center, :center),
        xlabel = "Dataset",
    )

    xs = Float64[]
    ncs = Float64[]
    mqs = Float64[]
    for row in eachrow(summary)
        dataset_offset = (levelcode(row.dataset) - 1) * (n_versions + 1)
        version_offset = levelcode(row.version) - 1
        push!(xs, dataset_offset + version_offset + 1)
        push!(ncs, row.nc_mean)
        push!(mqs, row.mq_mean)
    end
    for (index, label) in enumerate(levels(summary.version))
        indices = ((index - 1) * n_datasets + 1):(index * n_datasets)
        for (ys, alpha) in [(mqs, 0.6), (ncs, 1.0)]
            barplot!(
                ax_accuracy,
                xs[indices],
                ys[indices];
                color = colors[index],
                alpha = alpha,
                width = 1,
                label = alpha == 1 ? label : nothing,
            )
        end
    end
end

# Plot timing
let
    global ax_runtime = Axis(
        grid[1, 2],
        title = "Runtime",
        limits = (nothing, nothing, 0, nothing),
        ylabel = "Runtime (all datasets, hours)",
        yticks = 1:11,
        xticksvisible = false,
        xticklabelsvisible = false,
    )

    sort!(combined, [:version]; by = levelcode)
    for i in 1:n_versions
        barplot!(
            ax_runtime,
            [i],
            [combined.hours_total[i]],
            color = colors[i],
            width = 1,
        )
    end
end

colgap!(grid, 10)
colsize!(grid, 2, Relative(0.2))

axislegend(ax_accuracy, "Vamb version", position = :cb)

save("benchmark.png", fig)
