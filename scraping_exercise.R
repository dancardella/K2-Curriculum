# A function that respectively: i) reads in preliminary stock symbols ii) crawls Yahoo Finance and downloads daily stock price data iii) builds data tables and iv) runs stock price correlations
# Secondary_Sample_List_Backtest
the_whole_enchillada <- function()
  {
  setwd("/Users/dancardella/Desktop/Wellstream/Secondary_Event/Stock_Prices")
  symbols <- read.csv("Secondaries_Sample_List.csv", header = TRUE)
  num_of_stocks<- seq(1, length(symbols[,1]))

  update_prices <- function()
    {
  for (row in num_of_stocks){
    stock<- symbols[row,1]
    file_name<- paste("http://chart.finance.yahoo.com/table.csv?s=", stock,"&a=9&b=30&c=2015&d=9&e=30&f=2016&g=d&ignore=.csv", sep="")
    download.file(file_name, paste("./",stock,"prices.csv", sep=""),method ="curl")
  }
  list.files ("./")
}
  build_correlation_tables <- function()
    {
      f <- read.csv("WGPprices.csv",header = TRUE)
      f_date_col<- f[,1]
      volumes_table <- data.frame(f_date_col)
      stock_names =list("Date")
      prices_table <- data.frame(f_date_col)
      
    for (row in num_of_stocks)
      {
      stock<- symbols[row,1]
      f <- read.csv(paste(stock,"prices.csv", sep=""),header = TRUE)
      prices_col <- f[,7]
      #volumes_col <- f[,6]
      prices_table <- cbind(prices_table,prices_col)
      #volumes_table <- cbind(volumes_table,volumes_col)
      stock<- as.vector(stock, mode="character")
      stock_names <- c(stock_names, stock)
      } 
      colnames(prices_table) <- stock_names
      #colnames(volumes_table) <- stock_names
      print(prices_table)
      #print(volumes_table)
      write.csv(prices_table, "Secondary_Sample_List_Backtest_Prices_Table.csv", 
                 row.names = FALSE, fileEncoding = "utf8")
      # write.csv(volumes_table, "Volumes_table.csv", 
      #           row.names = FALSE, fileEncoding = "utf8")
    
    
      prices_table<- prices_table[,-1]
      # cor_table houses all correlations ***
      cor_table<- cor(prices_table)
      
      pairs_table <- c()
      
      symbols_base_stock <- read.csv("Secondaries_Sample_List.csv", header = FALSE)
      num_of_stocks_base_stock<- seq(1, length(symbols_base_stock[,1]))
      
      for (i in (num_of_stocks)-1)
      {
        bool<- cor_table[i,]<1
        index<- which.max(cor_table[i,bool])
        stock_pair<- names(cor_table[i,bool][index])
        base_stock<- rownames(cor_table)[i]
        pairs_table[base_stock] <- stock_pair
      }
      
      #print(pairs_table)
      #print(cor_table)
      write.csv(cor_table, "Secondary_Sample_List_Backtest_Correlation_Table.csv", 
                row.names = TRUE, fileEncoding = "utf8")
      write.csv(pairs_table, "Secondary_Sample_List_Backtest_Pairs_Table.csv", 
                row.names = TRUE, fileEncoding = "utf8")
  }
  update_prices()
  build_correlation_tables()
  
}

the_whole_enchillada()






