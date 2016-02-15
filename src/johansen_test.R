library("urca") # For cointegration 
#library("stats")
library("tseries")
library("MTS")

setwd("C:/Users/Mark/Documents/sr year/sociology")

extract_results = function(jtest) {
  out = data.frame(jtest@cval)
  out['test'] = jtest@teststat
  return(out)
}

delta = function(series, omit=NULL) {
  delta_series = tail(series,-1)-head(series,-1)
  if(!is.null(omit)){
    for(colname in omit){
      delta_series$colname = tail(series$colname,-1)
    }
  }
  return(delta_series)
}

clean_cols = function(df) return(df[, colSums(is.na(df)) == 0])

remove_zero_cols = function(df) return(df[, colSums(abs(df)) > 0])

clean_rows = function(df) return(df[rowSums(is.na(df)) == 0,])

adf_pvalue = function(series) return(adf.test(series)$p.value)

helper_pass_jahansen = function(series) {
  jtest = ca.jo(series,type="trace",K=2,season=NULL,dumvar=NULL,spec="longrun",ecdet = "trend")
  res = extract_results(jtest)
  pass = sum(res$test<res$X1pct)==0
  return(pass)
}

split_by_cols= function(series,max_cols=10) {
  if (ncol(series)>max_cols){
    nvar = floor(ncol(series)/2)
    dataframes = list(series[,1:nvar],series[,nvar:ncol(series)])
    result_list  = lapply(dataframes,split_by_cols)
    return(unlist(result_list, recursive=FALSE))
  }else{
    return(list(series))
  }
}

pass_johansen = function(series) {
  series = subset(series, select = -c(year))
  series_rowclean = remove_zero_cols(clean_rows(series))
  series_colclean = remove_zero_cols(clean_cols(series))
  
  dataframes = unlist(list(split_by_cols(series_rowclean),split_by_cols(series_colclean)),recursive=FALSE)
  
  #Johansen cointegration test on all datasets
  results = sapply(dataframes,helper_pass_jahansen)
  
  #return false if any of the dataframes failed the jh test
  return(sum(!results)==0)
}

pass_adf = function(series,p_val=.01) {
  series = subset(series, select = -c(year))
  series_rowclean = remove_zero_cols(clean_rows(series))
  series_colclean = remove_zero_cols(clean_cols(series))
  
  results_row = sapply(series_rowclean, adf_pvalue)
  results_col = sapply(series_colclean, adf_pvalue)
  fail_row = sum(results_row>p_val)
  fail_col = sum(results_col>p_val)
  return(fail_row==0 & fail_col==0)
}

#Normal data
series = read.csv('data/Econtext Time Series with merchant info.csv',header=T, na.strings='.')
series_categorical = c('disruptions', 'domestic politics')
series_delta = delta(series, omit=series_categorical)

#Data with lags and interactions amongst non categorical variables
enriched_series = read.csv('data/enriched_data.csv')
enriched_series_categorical = c(series_categorical, c('disruptions_l1', 'domestic politics_l1'))
enriched_series_delta = delta(enriched_series, omit=enriched_series_categorical)

#Print and run results
base_string = "Johansen test for %s succeded: %s"
print(sprintf(base_string, "series",toString(pass_johansen(series))))
print(sprintf(base_string, "series_delta", toString(pass_johansen(series_delta))))
print(sprintf(base_string, "enriched_series", toString(pass_johansen(enriched_series))))
print(sprintf(base_string, "enriched_series_delta", toString(pass_johansen(enriched_series_delta))))

#Deeper investigation of VECM
series = subset(series, select = -c(year))
series_rowclean = remove_zero_cols(clean_rows(series))
series_colclean = remove_zero_cols(clean_cols(series))
jtest = ca.jo(series,type="trace",K=2,season=NULL,dumvar=NULL,spec="longrun",ecdet = "trend")
#print(summary(jtest))


m1=VARMA(series_colclean,p=1,q=1,include.mean=FALSE)