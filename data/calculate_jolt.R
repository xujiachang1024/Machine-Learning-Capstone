calculate_jolt <- function(fileName, output) {
    
    # read .csv file
    df <- read.csv(fileName)
    
    # calculate x-axis jolt
    df$x_jolt = 0.00;
    for (i in 2:nrow(df)) {
        df$x_jolt[i] <- df$x[i] - df$x[i-1]
    }
    
    # calculate y-axis jolt
    df$y_jolt = 0.00;
    for (i in 2:nrow(df)) {
        df$y_jolt[i] <- df$y[i] - df$y[i-1]
    }
    
    # calculate z-axis jolt
    df$z_jolt = 0.00;
    for (i in 2:nrow(df)) {
        df$z_jolt[i] <- df$z[i] - df$z[i-1]
    }
    
    # write to a new "speedbumps.csv" file
    write.csv(df, file = output)
}