find_speedbump <- function(fileName, output) {
    
    # read .csv file
    df <- read.csv(fileName)
    
    # calculate z-axis jolt
    df$z_jolt = 0.00;
    for (i in 2:nrow(df)) {
        df$z_jolt[i] <- df$Z[i] - df$Z[i-1]
    }
    
    # calculate mean and standard deviation of the $z_jolt column
    as.numeric(df$z_jolt)
    z_jolt.mean = mean(df$z_jolt);
    z_jolt.sdev = sd(df$z_jolt);
    print(summary(df$z_jolt))
    print(z_jolt.sdev)
    
    # find speedbumps
    df$speedbump <- NA
    for (i in 1:nrow(df)) {
        if (df$z_jolt[i] <= z_jolt.mean - 5 * z_jolt.sdev | df$z_jolt[i] >= z_jolt.mean + 5 * z_jolt.sdev) {
            df$speedbump[i] <- "yes"
        }
        else {
            df$speedbump[i] <- "no"
        }
    }
    
    # factor
    as.factor(df$speedbump)
    
    speedbumps <- df[df$speedbump == "yes", ]
    print(nrow((speedbumps)))
    
    # write to a new "speedbumps.csv" file
    write.csv(df, file = output)
}