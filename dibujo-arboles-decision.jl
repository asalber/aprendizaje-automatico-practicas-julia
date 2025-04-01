
```{julia}
using MLJ, OpenML, DataFrames, MLJDecisionTreeInterface, ScientificTypes , Graphs, GraphMakie, NetworkLayout, GLMakie

import DataFrames as DF

import DecisionTree: DecisionTreeClassifier,Leaf,Node,Root,depth,InfoNode

function Base.convert(::Type{SimpleDiGraph}, model::InfoNode; maxdepth = depth(model))

	if maxdepth == -1
		maxdepth = depth(model.node)
	end
	g = SimpleDiGraph()
	properties = Any[]
	features = model.info.featurenames
	walk_tree!(model.node, g, maxdepth, properties, features)
	return g, properties
end

Base.convert(::Type{SimpleDiGraph}, model::DecisionTreeClassifier; kwargs...) =
	Base.convert(SimpleDiGraph, model.node; kwargs...)

function walk_tree!(node::Node, g, depthLeft, properties, features)

	add_vertex!(g)

	if depthLeft == 0
		push!(properties, (Nothing, "..."))
		return vertices(g)[end]
	else
		depthLeft -= 1
	end

	current_vertex = vertices(g)[end]
	val = node.featval

	featval = isa(val, AbstractString) ? val : round(val; sigdigits = 2)
	featurename = features[node.featid]
	label_node = (Node, "$(featurename) < $featval ?")
	push!(properties, label_node)

	child = walk_tree!(node.left, g, depthLeft, properties, features)
	add_edge!(g, current_vertex, child)

	child = walk_tree!(node.right, g, depthLeft, properties, features)
	add_edge!(g, current_vertex, child)

	return current_vertex
end

function walk_tree!(leaf::Leaf, g, depthLeft, properties, features)
	add_vertex!(g)
	n_matches = count(leaf.values .== leaf.majority)
	#ratio = string(n_matches, "/", length(leaf.values))

	emojis_class = Dict("1" => "😊", "2" => " ☹️")
	leaf_class = emojis_class[string.(leaf.majority)]

	push!(properties, (Leaf, "$(leaf_class)"))# : $(ratio)"))
	return vertices(g)[end]
end

```


```{julia}
function GraphMakie.graphplot(model::Union{InfoNode,DecisionTreeClassifier}; kwargs...)
	f, ax, h = plotdecisiontree(model; kwargs...)
	hidedecorations!(ax)
	hidespines!(ax)
	ax.aspect = DataAspect()
	return f
end

@recipe(PlotDecisionTree) do scene
		Attributes(
			nodecolormap = :darktest,
			textcolor = RGBf(0.5,0.5,0.5),
			leafcolor = :darkgreen,
			nodecolor = :white,
			maxdepth = -1,
		)
end
	
function Makie.plot!(
		plt::PlotDecisionTree{<:Tuple{<:Union{InfoNode,DecisionTreeClassifier}}},
	)
	
		@extract plt leafcolor, textcolor, nodecolormap, nodecolor, maxdepth
		model = plt[1]
	
		# convert to graph
		tmpObs = @lift convert(SimpleDiGraph, $model; maxdepth = $maxdepth)
		graph = @lift $tmpObs[1]
		properties = @lift $tmpObs[2]
	
		# extract labels
		labels = @lift [string(p[2]) for p in $properties]
	
		# set the colors, first for nodes & cutoff-nodes, then for leaves
		nlabels_color = map(
			properties,
			labels,
			leafcolor,
			textcolor,
			nodecolormap,
		) do properties, labels, leafcolor, textcolor, nodecolormap
	
			# set colors for the individual elements
			leaf_ix = findall([p[1] == Leaf for p in properties])
			leafValues = [p[1] for p in split.(labels[leaf_ix], " : ")]
	
			# one color per category
			uniqueLeafValues = unique(leafValues)
			individual_leaf_colors =
				resample_cmap(nodecolormap, length(uniqueLeafValues))
			nlabels_color =
				Any[p[1] == Node ? textcolor : leafcolor for p in properties]
			for (ix, uLV) in enumerate(uniqueLeafValues)
				ixV = leafValues .== uLV
				nlabels_color[leaf_ix[ixV]] .= individual_leaf_colors[ix]
			end
			return nlabels_color
		end
	
		fontsize = @lift .-length.($labels) .* 0.1 .+ 14
		graphplot!(
			plt,
			graph;
			layout = Buchheim(),
			nlabels = labels,
			#nlabels_distance=10,
			node_size = 80,
			node_color = nodecolor,
			nlabels_color = nlabels_color,
			nlabels_fontsize = fontsize,
			nlabels_align = (:center, :center),
			#tangents=((0,-1),(0,-1))
		)
		return plt
end
	
recipe_defined = true

import GLMakie: @recipe
```


```{julia}
task = OpenML.load(287) 
data = DF.DataFrame(task)
DF.transform!(data, :quality => (x -> ifelse.(x .> 6.5, "good", "bad")) => :quality)
data[!, :quality] = categorical(data[!, :quality])

y, X = unpack(data, ==(:quality); rng=123);
train, test = partition(eachindex(y), 0.8)

Tree = @load DecisionTreeClassifier pkg = "DecisionTree" verbosity = false
model = Tree(max_depth=4, min_samples_leaf=220)

mach = machine(model, X, y)
mach = MLJ.fit!(mach; rows=train)
tree_params = fitted_params(mach)

features = [(i,tree_params.features[i]) for i in 1:length(tree_params.features)]
g = GraphMakie.graphplot(tree_params.tree; nlabels=tree_params.tree.info.featurenames)
```


using MLJ
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model = DecisionTreeClassifier(max_depth=3, min_samples_split=3)

X, y = @load_iris
mach = machine(model, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) ## probabilistic predictions
predict_mode(mach, Xnew)   ## point predictions
pdf.(yhat, "virginica")    ## probabilities for the "verginica" class

tree = fitted_params(mach).tree

using Plots, TreeRecipe
Plots.plot(tree) ## for a graphical representation of the tree

feature_importances(mach)