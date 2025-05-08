# save as stats.awk

# awk -f stats.awk file1.txt file2.txt file3.txt > result.txt
# awk -f stats.awk dat_specheat_rex_000000* > dat_specheat_rex

function abs(x){return (x>0)? x:-x}

{
    sum1[FNR] += $1
    sum2[FNR] += $2
    sumsq1[FNR] += $1 * $1
    sumsq2[FNR] += $2 * $2

    count[FNR]++

    if (FNR > max_line) {
        max_line = FNR
    }
}

END {
    for (i = 1; i <= max_line; i++) {
        n = count[i]
        if (n > 0) {
            mean1 = sum1[i] / n
            mean2 = sum2[i] / n

            stddev1 = sqrt(abs(sumsq1[i] / n - mean1 * mean1))
#            stddev2 = sqrt(abs(sumsq2[i] / n - mean2 * mean2))
            stddev2 = sqrt(abs(sumsq2[i] / n - mean2 * mean2) / (n - 1))

            # output: col1_mean col1_stddev col2_mean col2_stddev
#            printf "%.6f %.6f %.6f %.6f\n", mean1, stddev1, mean2, stddev2
            printf "%.6f %.6f %.6f\n", mean1, mean2, stddev2
        } else {
            # output empty data if there is no row
#            print "0 0 0 0"
            print "0 0 0"
        }
    }
}

