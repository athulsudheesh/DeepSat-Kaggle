using CSV
using DataFrames
using Dates
using Plots
plotly()

df = CSV.read("gokul.csv",DataFrame)
df.dateTime = DateTime.(df.dateTime,"yyyy-mm-dd HH:MM:SS")


hs = [Plots.plot(df.dateTime, df[:,i], title=names(df)[i]) for i in 4:n]
ncols = 5
nrows = cld(length(hs), ncols)
blankplot = Plots.plot(legend=false,grid=false,foreground_color_subplot=:white)
for i = (length(hs)+1):(nrows*ncols)
	push!(hs, deepcopy(blankplot))
end
final = Plots.plot(hs..., fmt=:png, layout=(nrows, ncols), legend=false, size=(3000,1500))

Plots.savefig(final,"final.png")