set terminal svg enhanced size 3000 2000 fsize 32
set print "-"

set xlabel "Iterations"
set ylabel "Time (s)"
set yrange[0.000010:0.000500]
set format y "%.1te%+03T";

set output "pfe.svg"
set title "PFE plot"
stats "./pfe.dat" using 2 prefix "A"
plot "./pfe.dat" using 1:2 title "", A_mean title gprintf("Mean = %.5te%+03T s", A_mean)

set output "grid_launch.svg"
set title "Grid Launch plot"
stats "./grid_launch.dat" using 2 prefix "A"
plot "./grid_launch.dat" using 1:2 title "", A_mean title gprintf("Mean = %.5te%+03T s", A_mean)
