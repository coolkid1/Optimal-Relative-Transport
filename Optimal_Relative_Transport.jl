using DataFrames, GLM, StatsKit, Random, Plots, StatsBase, Combinatorics, Logging, Flux #OptimalTransport
import OptimalTransport 
ot = OptimalTransport

#directory for Images
output_dir = "/Users/"

using LinearAlgebra
# here we randomly generate random examples from two hypothetical distributions
Random.seed!(123)
n = 10^3

# Loop through all permutations of the parameters
# TODO: write a genereic injest data that will ouput these relative distributions via a Database
dists = [Normal, Gamma, Beta, LogNormal, Laplace, Weibull]
params = [[1, 1], [1, 2], [2, 1], [2, 2]]#[[1, 1], [1, 2], [2, 1], [2, 2]]
# Seeing the cost and maps for each dataframe
df_loss = DataFrame(Distribution_1 = String[], Distribution_2 = String[], Cost = Float64[])
OTPan_array = Matrix{Float64}[]
for (i, dist1) in enumerate(dists)
    for (j, dist2) in enumerate(dists)
        for param1 in params
            for param2 in params
                x = rand(dist1(param1[1], param1[2]), n) #setting each distributions 
                y = rand(dist2(param2[1], param2[2]), n)
                println("Pair ($i,$j): ($dist1($(param1[1]),$(param1[2])), $dist2($(param2[1]),$(param2[2])))")

                #Target is completly uniform
                bins = collect(range(0, stop=1, length=n+1))  # generate n+1 evenly spaced bins
                target = (bins[1:end-1] .+ bins[2:end]) / 2  #rand(d3, n) #generate n uniformly spaced values in each bin

                df = DataFrame(X = x, Y = y, Target = target)

                
                # the cdf of any given distributions
                #n = nrow(df)
                for i in names(df)
                    s = "Empirical Cumluative Distribution of " * i
                    p = plot(sort(df[!,i]), (1:n)./n, 
                        xlabel = "sample", ylabel = "Probability", 
                        title = s, label = "")
                    #display(p)

                    sort_val = sort(df[!,i]) 
                    cdf_val = (1:n)./n
                    var_sort = i * "_sorted"
                    var_cdf = i * "_cdf"
                    df[!, var_sort] = sort_val
                    df[!, var_cdf] = cdf_val
                end



                # Empirical estimation of CDF and PDF
                ## Empirical CDF
                # """
                # This is from StatsBase and how the code is ordering for an emperical CDF
                #     ecdf(X)
                # Return an empirical cumulative distribution function (ECDF) based on a vector of samples
                # given in `X`.
                # Note: this is a higher-level function that returns a function, which can then be applied
                # to evaluate CDF values on other samples.

                # Getting the CDF for out, as we can see here, we get a uniform plot
                F = ecdf(df.X);

                #here we will try to extract the histogram feature to build a KDE
                #for more detail on how to use the KDE follow the below website
                #https://github.com/JuliaStats/KernelDensity.jl 

                #the densities of the two distributions
                k_1 = kde(df.X)
                k_2 = kde(df.Y)
                k_target = kde(df.Target)

                #the relative density
                R = F(df.Y);
                function threshold_to_one(vec::AbstractVector{T}, precision::Int=16) where T<:Real
                    """
                    Some Values in R get lumped into 1, so we push it back 1 bin to adjust R
                    1. create a copy of the input vector
                    2. set values greater than or equal to 1 to 0.999
                    3. copy the values less than 1 to the output vector
                    """
                    vec_out = similar(vec)
                    vec_out[vec .>= 1.] .= 0.9999999
                    vec_out[vec .< 1.] = vec[vec .< 1.]
                    return vec_out
                end
                R = threshold_to_one(R)
                df[!, "relative_dist"] = R 
                k_3 = kde(R)

                #the two densities against each other and no the relative
                h1 = histogram([df.X, df.Y], normalize = true, labels = ["$dist1($(param1[1]),$(param1[2]))" "$dist2($(param2[1]),$(param2[2]))"]
                ,title = " $dist1($(param1[1]),$(param1[2])) and $dist2($(param2[1]),$(param2[2]))) Distribution", size = (800, 800))
                plot!([k_1.x, k_2.x], [k_1.density, k_2.density]
                    ,linewidth = 3, labels = labels = ["$dist1($(param1[1]),$(param1[2]))" "$dist2($(param2[1]),$(param2[2]))"]) # plotting with X as the reference
                display(h1)
                savefig(h1, joinpath(output_dir,"$dist1($(param1[1]),$(param1[2])), $dist2($(param2[1]),$(param2[2]))) Distribution.png"))


                h1 = histogram([df.relative_dist, df.Target], normalize = true, 
                labels = ["df.relative_dist" "df.Target"], title = "Relative Distribution of of $dist1($(param1[1]),$(param1[2])), $dist2($(param2[1]),$(param2[2]))) and Uniform", size = (800, 800))

                plot!([k_3.x, k_target.x], [k_3.density, k_target.density]
                , xlims = (0,1),linewidth = 3,labels = ["df.relative_dist" "df.Target"]) 
                #hline!([1], linestyle=:dash, linewidth = 3, )
                display(h1)
                savefig(h1 ,joinpath(output_dir,"Relative Distribution of $dist1($(param1[1]),$(param1[2])), $dist2($(param2[1]),$(param2[2]))) and Unifrom.png"))

                ### using some packages to find the OTP from RD to the uniform RD

                ##################
                #OptimalTransport#
                ##################
                p  = fill(1/n, n)#normalize!(df.relative_dist)
                # our Target at the moment is the unifomrm
                q = fill(1/n, n)#normalize!(df.Target) 

                #check if the source and target marginal are balanced
                ot.checkbalanced(p,q)

                C = pairwise(SqEuclidean(), df.relative_dist', df.Target'; dims=2);
                # random cost matrix
                #C = pairwise(SqEuclidean(), rand(1,size(p,1)), rand(1,size(q,1)); dims=2)
                # C = zeros(n, n)
                # for i in 1:n
                #     for j in 1:n
                #         C[i, j] = (df.relative_dist[i]- df.Target[j])^2
                #         #C[i, j] = abs(df.relative_dist[i] - df.Target[j])^3
                #     end            
                # end

                #check size of C is balanced with p and q
                #ot.checksize(p,q,C)

                #ot_cost(C, p, q)
                # # regularization parameter
                # # if we increse ε, the closer the plan moves to the other distributions
                #ε = 0.01

                # # solve entropically regularized optimal transport problem
                # # increasing n makes the otp too costly
                #ot_plan = sinkhorn2(p, q, C, ε, regularization=true)

                # # cost of the plan
                # cost = sum(ot_plan .* C)

                #looping through our regulizer to get a detailed plan                
                ε = 0.01 # our entroic regulizer
                prev_ot_plan = ot_plan = zeros(n, n) #maybe save the convergence plot
                while !any(isnan.(ot_plan))
                    prev_ot_plan = ot_plan
                    ε /= 2
                    print(ε)
                    ot_plan = sinkhorn(p, q, C, ε, maxiter=10_000, rtol = 1e-9);
                end;

                ot_plan = prev_ot_plan
                
                # push cost of transformation into a dataframe
                cost = sum(ot_plan .* C)
                push!(df_loss, (Distribution_1 = "$dist1($(param1[1]),$(param1[2]))", 
                Distribution_2 = "$dist2($(param2[1]),$(param2[2]))", Cost = cost))
                
                #pushing the ot_plan to see the avg
                push!(OTPan_array, ot_plan)
                
                h1 = heatmap(C,title="Cost Matrix of $dist1($(param1[1]),$(param1[2])), $dist2($(param2[1]),$(param2[2])))", size = (800, 800))
                display(h1)
                savefig(joinpath(output_dir,"Heatmap of Cost Matrix of $dist1($(param1[1]),$(param1[2])), $dist2($(param2[1]),$(param2[2]))).png"))

                h1 = heatmap((ot_plan),title="OT Plan of $dist1($(param1[1]),$(param1[2])), $dist2($(param2[1]),$(param2[2])))", size = (800, 800))
                display(h1)
                savefig(h1 ,joinpath(output_dir,"Heatmap of Optimal Transport Plan of $dist1($(param1[1]),$(param1[2])), $dist2($(param2[1]),$(param2[2]))).png"))

            end
        end
    end
end

#plot(df_loss.Cost)
#mean(df_loss.Cost)
#########################
# The different Heatmaps#                 
#########################
h1 = heatmap(unique(df_loss.Distribution_1), unique(df_loss.Distribution_2,), 
reshape(df_loss.Cost, (length(unique(df_loss.Distribution_1)), length(unique(df_loss.Distribution_2)))),
        xlabel = "Dist 1", ylabel = "Dist 2", colorbar_title = "Cost", aspect_ratio = :equal, size = (1000, 1000),
        xticks = (1:length(unique(df_loss.Distribution_1)), unique(df_loss.Distribution_1)), yticks = (1:length(unique(df_loss.Distribution_2,)), unique(df_loss.Distribution_2,)),
        xrotation = -90,
        title = "Heatmap of Each Distributions Total Cost")
        #xticks!(1:10, x_labels, rotation = 90)
display(h1)
savefig(joinpath(output_dir,"Heatmap of Each Distributions Total Cost.png"))
# Calculate the average of each element
#OTPan_avg = mean(OTPan_array)#[mean(OTPan_array[i]) for i in 1:length(OTPan_array)]
#h1 = heatmap((OTPan_avg),title="Element Avg for Every Optimal Transport Plan")
#display(h1)
#savefig(h1 ,joinpath(output_dir,"Element Avg for Every Optimal Transport Plan.png"))


df_loss_kl = DataFrame(Distribution_1 = String[], Distribution_2 = String[], KL_div = Float64[])
for (i, dist1) in enumerate(dists)
    for (j, dist2) in enumerate(dists)
        for param1 in params
            for param2 in params
                x = rand(dist1(param1[1], param1[2]), n).+ 30
                y = rand(dist2(param2[1], param2[2]), n).+ 30
                x /= sum(x)
                y /= sum(y)
                # Cal KL divergence
                kl_div = kldivergence(Categorical(x), Categorical(y))

                push!(df_loss_kl, (Distribution_1 = "$dist1($(param1[1]),$(param1[2]))", 
                Distribution_2 = "$dist2($(param2[1]),$(param2[2]))", KL_div = kl_div))
            end
        end
    end
end                



h1=heatmap(unique(df_loss_kl.Distribution_1), unique(df_loss_kl.Distribution_2,), 
reshape(df_loss_kl.KL_div, (length(unique(df_loss_kl.Distribution_1)), length(unique(df_loss_kl.Distribution_2)))),
        xlabel = "Dist 1", ylabel = "Dist 2", colorbar_title = "KL_div", aspect_ratio = :equal, size = (1000, 1000),
        xticks = (1:length(unique(df_loss_kl.Distribution_1)), unique(df_loss_kl.Distribution_1)),yticks = (1:length(unique(df_loss_kl.Distribution_2,)), unique(df_loss_kl.Distribution_2,)),
        xrotation = -90,
        title = "Heatmap of Each Distributions KL Diveregence")
        #xticks!(1:10, x_labels, rotation = 90)
savefig(joinpath(output_dir,"Heatmap of Each Distributions Each Distributions KL Diveregence.png"))



#2-Wasserstein distance
# a squareroot distance
df_loss_ws = DataFrame(Distribution_1 = String[], Distribution_2 = String[], Ws = Float64[])
for (i, dist1) in enumerate(dists)
    for (j, dist2) in enumerate(dists)
        for param1 in params
            for param2 in params
                x = dist1(param1[1], param1[2])               
                y = dist2(param2[1], param2[2]) 
                #c(x, y) = (abs(x - y))^2 # could have used `sqeuclidean` from `Distances.jl`
                print("$dist1($(param1[1]),$(param1[2]))", "$dist2($(param2[1]),$(param2[2]))")
                # Cal Wasserstein
                ws=[]
                
                try #NaNs when no Wasserstein is calculated
                    ws = wasserstein(x, y; p=2)
                catch
                    ws = NaN64
                end
                print(ws)

                push!(df_loss_ws, (Distribution_1 = "$dist1($(param1[1]),$(param1[2]))", 
                Distribution_2 = "$dist2($(param2[1]),$(param2[2]))", Ws = ws))
            end
        end
    end
end                

h1=heatmap(unique(df_loss_ws.Distribution_1), unique(df_loss_ws.Distribution_2,), 
reshape(df_loss_ws.Ws, (length(unique(df_loss_ws.Distribution_1)), length(unique(df_loss_ws.Distribution_2)))),
        xlabel = "Dist 1", ylabel = "Dist 2", colorbar_title = "Wasserstein Distance", aspect_ratio = :equal, size = (1000, 1000),
        xticks = (1:length(unique(df_loss_ws.Distribution_1)), unique(df_loss_ws.Distribution_1)),yticks = (1:length(unique(df_loss_ws.Distribution_2,)), unique(df_loss_ws.Distribution_2,)),
        xrotation = -90,
        title = "Heatmap of Each Distributions Wasserstein Distance")
        #xticks!(1:10, x_labels, rotation = 90)
savefig(joinpath(output_dir,"Heatmap of Each Distributions Wasserstein Distance.png"))
