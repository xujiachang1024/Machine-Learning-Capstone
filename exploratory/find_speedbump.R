find_speedbump <- function(fileName) {
    
    # read .csv file
    df <- read.csv(fileName)
    
    # calculate z-axis jolt
    df$z.jolt = 0.00;
    for (i in 2:nrow(df)) {
        df$z.jolt[i] <- df$Z[i] - df$Z[i-1]
    }
    
    # calculate mean and standard deviation of the $z.jolt column
    as.numeric(df$z.jolt)
    z.jolt.mean = mean(df$z.jolt);
    z.jolt.sdev = sd(df$z.jolt);
    print(summary(df$z.jolt))
    print(z.jolt.sdev)
    
    # find speedbumps
    df$speedbump <- NA
    for (i in 1:nrow(df)) {
        if (df$z.jolt[i] <= z.jolt.mean - 5 * z.jolt.sdev | df$z.jolt[i] >= z.jolt.mean + 5 * z.jolt.sdev) {
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
    write.csv(df, file = "speedbumps.csv")
}