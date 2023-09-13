# module load tools
# module load julia/1.9.0
# export JULIA_DEPOT_PATH=/home/projects/cpr_10006/people/svekut/julia_packages
# julia --startup-file=no --project=@vambbench

using DataFrames, CSV
using VambBenchmarks


function get_nbins(binning::Binning, recall::Real, precision::Real; tax_level::Integer=0, assembly::Bool=false)::Integer
    ri = searchsortedfirst(binning.recalls, recall)
    ri > length(binning.recalls) && error("Binning did not benchmark at that high recall")
    pi = searchsortedfirst(binning.precisions, precision)
    pi > length(binning.precisions) && error("Binning did not benchmark at that high precision")
    matrices = assembly ? binning.recovered_asms : binning.recovered_genomes
    if tax_level + 1 ∉ eachindex(matrices)
        error(lazy"Requested bins at taxonomic level $tax_level but have only level 0:$(lastindex(matrices)-1)")
    end
    m = matrices[tax_level + 1]
    m[pi, ri]
end

result_df = DataFrame(folder=String[], genomes=Int[], assemblies=Int[], level=String[])

vars = ["Oral", "Skin", "Urogenital", "Airways", "Gastrointestinal"]
rec = 0.9
prec = 0.95
exp = "0.6"
for var in vars
    ref_file = "/home/projects/cpr_10006/people/paupie/vaevae/spades_ef_refs_/ref_spades_$(var).json"
    ref =  open(i -> Reference(i), ref_file)
    bins = gold_standard(ref)
    output_genomes = get_nbins(bins, rec, prec, tax_level=0, assembly=false)
    output_as = get_nbins(bins, rec, prec, tax_level=0, assembly=true)

    output_genomes_sp = get_nbins(bins, rec, prec, tax_level=1, assembly=false)
    output_as_sp = get_nbins(bins, rec, prec, tax_level=1, assembly=true)

    output_genomes_g = get_nbins(bins, rec, prec, tax_level=2, assembly=false)
    output_as_g = get_nbins(bins, rec, prec, tax_level=2, assembly=true)

    # Append the result to the DataFrame
    push!(result_df, ("gold_standard_$(var)", output_genomes, output_as, "Strain"))
    push!(result_df, ("gold_standard_$(var)", output_genomes_sp, output_as_sp, "Species"))
    push!(result_df, ("gold_standard_$(var)", output_genomes_g, output_as_g, "Genus"))
    for file_path in [
        "/home/projects/cpr_10006/people/svekut/cami2_$(var)_reassembled_$(exp)/vaevae_clusters.tsv", 
        "/home/projects/cpr_10006/people/svekut/cami2_$(var)_reclustering_reassembled_$(exp)/clusters_reclustered.tsv",
        "/home/projects/cpr_10006/people/paupie/vaevae/semibin2_ptracker/semibin2_$(var)_multy_easy_bin_070723/samples/postreclustered_clusters.tsv",
        "/home/projects/cpr_10006/people/paupie/vaevae/semibin2_ptracker/semibin2_$(var)_multy_easy_bin_070723/samples/prereclustered_clusters.tsv",
        "/home/projects/cpr_10006/people/svekut/cami2_$(var)_vamb_1/vae_clusters.tsv",
        "/home/projects/cpr_10006/people/svekut/cami2_$(var)_reclustering_vamb_1/clusters_reclustered.tsv",
        "/home/projects/cpr_10006/people/svekut/cami2_$(var)_vamb_2/vae_clusters.tsv",
        "/home/projects/cpr_10006/people/svekut/cami2_$(var)_reclustering_vamb_2/clusters_reclustered.tsv",
        ]
        # Check if the file exists before calling the function
        if isfile(file_path)
            bins = open(i -> Binning(i, ref), file_path)

            output_genomes = get_nbins(bins, rec, prec, tax_level=0, assembly=false)
            output_as = get_nbins(bins, rec, prec, tax_level=0, assembly=true)

            output_genomes_sp = get_nbins(bins, rec, prec, tax_level=1, assembly=false)
            output_as_sp = get_nbins(bins, rec, prec, tax_level=1, assembly=true)

            output_genomes_g = get_nbins(bins, rec, prec, tax_level=2, assembly=false)
            output_as_g = get_nbins(bins, rec, prec, tax_level=2, assembly=true)

            # Append the result to the DataFrame
            push!(result_df, (file_path, output_genomes, output_as, "Strain"))
            push!(result_df, (file_path, output_genomes_sp, output_as_sp, "Species"))
            push!(result_df, (file_path, output_genomes_g, output_as_g, "Genus"))
        else
            println("File $file_path not found")
        end
    end
end


CSV.write("/home/projects/cpr_10006/people/svekut/results_$(exp).csv", result_df)

vars = ["Oral", "Skin", "Urogenital", "Airways", "Gastrointestinal"]

for var in vars
    result_df = DataFrame(genome_id=String[], source_id=String[], scgs=Int[], total=Int[])
    println("$var")
    ref_file = "/home/projects/cpr_10006/people/paupie/vaevae/spades_ef_refs_/ref_spades_$(var).json"
    ref =  open(i -> Reference(i), ref_file)
    marker_path = "/home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/cami_reassembled_$(var).hmmout"
    df = CSV.File(marker_path, comment="#") |> DataFrame
    new_column = [split(string, ".")[2] for string in df[!, 1]]
    contigs = Set([split(string, "_")[1] for string in new_column])
    for genome in genomes(ref)
        for source in genome.sources
            seqs = Set([seq[1].name for seq in source.sequences])
            n = source.name
            println("$n")
            push!(result_df, (genome.name, source.name, length(intersect(seqs, contigs)), length(seqs)))
        end
    end
    CSV.write("/home/projects/cpr_10006/people/svekut/scgs_$(var).csv", result_df)
end


for var in vars
    result_df = DataFrame(genome_id=String[], source_id=String[], scgs=Int[], total=Int[])
    println("$var")
    ref_file = "/home/projects/cpr_10006/people/paupie/vaevae/spades_ef_refs_/ref_spades_$(var).json"
    ref =  open(i -> Reference(i), ref_file)
    VambBenchmarks.subset!(ref; genomes=g -> Flags.organism in flags(g))
    res = Array(Float64, n)
    for i in 1:9
        c = top_clade(ref)

