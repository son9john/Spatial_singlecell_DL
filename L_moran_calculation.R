library(hash)
# library(RColorBrewer) #to use brewer.pal
# library(fields) #to use designer.colors
source("~/BSC/test_morani.R") ##@## get Hexagon, for debugging

# Function that input test data format and write Visium-style positions file
write_positions_data = function(test_position_data, spatial=spatial) {
  data = test_position_data
  for (i in 0:(nrow(data)-1)) { # i:y
    if (i %% 2 == 0) col_ixs = seq(0, ncol(data)-1, 2)
    else col_ixs = seq(1, ncol(data)-1, 2)
    
    for (j in col_ixs) { # j:x
      value = data[i+1, j+1]
      if (is.na(value)) next
      if (value == 0) in_tissue = 0
      else { in_tissue = 1 }
      
      barcode = sprintf("i%d_j%d", i, j) # i:y, j:x
      xcoord = 10*(i+1) # i:y
      ycoord = 10*(j+1) # j:x
      
      print_str = sprintf("%s,%d,%d,%d,%d,%d", barcode, in_tissue, 
                          i, j, 10*i, 10*j)
      write(print_str, spatial$positions_file, append = T)
      
      # if (value == 0) color = "yellow"
      # else color = colors[value]
      # Hexagon(j+1, nrow(data)-(i+1), col=color) # x, y - draw hexagon
    }
  }
}

# Visium-style spatial file settings
Spatial = function(positions_path) {
  spatial = list()
  spatial$positions_path = positions_path
  spatial$positions_file = file(spatial$positions_path, 'w')
  return(spatial)
}

# Visium-style file settings
Visium = function(out_dir="visium/") {
  visium = list() # data holder
  visium$barcodes_path = paste(out_dir, "barcodes.tsv.gz", sep='') # AAACAAGTATCTCCCA-1, ...
  visium$features_path = paste(out_dir, "features.tsv.gz", sep='') # ENSMUSG00000051951  Xkr4  Gene  Expression, ...
  visium$matrix_path = paste(out_dir, "matrix.mtx.gz", sep='') # as below
  # %%MatrixMarket matrix coordinate integer general
  # %metadata_json: {"software_version": "spaceranger-1.1.0", "format_version": 2}
  # 32285 2695 15850802 # genes barcodes valuesCnt
  # 9 1 2
  # 11 1 2
  visium$barcodes_file = gzfile(visium$barcodes_path, 'w')
  visium$features_file = gzfile(visium$features_path, 'w')
  visium$matrix_file = gzfile(visium$matrix_path, 'w')
  return(visium)
}

# Data per gene, given as test data format
GeneTestData = function(gsym, in_dir="~/local/lees2001/") {
  gene = list()
  test_data_path = paste(in_dir, gsym, '.graph.txt', sep='')
  if (!file.exists(test_data_path)) write(paste("[ERROR]", test_data_path, "does not exist."), stderr())
  test_data = read.table(test_data_path, 
                         na.strings=".", sep=" ", row.names=1, header=T)
  return(test_data)
}

# Function that input test data format and write Visium-style output files
#  input: gsyms = ['A', 'B']
#  output: barcodes.tsv, features.tsv, matrix.mtx
write_visium_data = function(gsyms, visium=visium) {
  visium$barcode2bid = hash()
  visium$bid2barcode = hash()
  visium$feature2fid = hash()
  visium$fid2feature = hash()
  barcode_cnt = 0
  feature_cnt = 0
  
  # Set and save features (gene info)
  gene_test_data = list()
  for (gsym in gsyms) {
    gene_test_data[[gsym]] = GeneTestData(gsym)
  
    # ENSMUSG00000051951  Xkr4  Gene  Expression, ...
    gene_info = paste(gsym, gsym, "Gene", "Expression", sep="\t")
    write(gene_info, visium$features_file, append=T)
    feature_cnt = feature_cnt + 1
    visium$feature2fid[gsym] = feature_cnt
    visium$fid2feature[toString(feature_cnt)] = gsym
  }
  nrows = nrow(gene_test_data[[gsym]])
  ncols = ncol(gene_test_data[[gsym]])
  
  for (i in 0:(nrows-1)) { # i:y
    if (i %% 2 == 0) col_ixs = seq(0, ncols-1, 2)
    else col_ixs = seq(1, ncols-1, 2)
  
    for (j in col_ixs) { # j:x
  
      # Set and save barcode
      barcode = sprintf("i%d_j%d", i, j) # i:y, j:x
      # AAACAAGTATCTCCCA-1, ...
      write(barcode, visium$barcodes_file, append=T)
      barcode_cnt = barcode_cnt + 1
      barcode_id = toString(barcode_cnt)
      visium$barcode2bid[barcode] = barcode_cnt
      visium$bid2barcode[barcode_id] = barcode
  
      for (gsym in gsyms) {
        fid = visium$feature2fid[[gsym]] # gene ID
        value = gene_test_data[[gsym]][i+1, j+1]
        # print(value) ##@##
        if (is.na(value)) next
        else if (value == 0) in_tissue = 0
        else in_tissue = 1
  
        # %%MatrixMarket matrix coordinate integer general
        # %metadata_json: {"software_version": "spaceranger-1.1.0", "format_version": 2}
        # 32285 2695 15850802 # genes barcodes valuesCnt
        # 9 1 2
        # 11 1 2
        if (in_tissue == 0) next # discard line if not in tissue
        mtx_line = paste(fid, barcode_id, value, sep=" ")
        write(mtx_line, visium$matrix_file, append=T)
      }
    }
  }
  close(visium$barcodes_file)
  close(visium$features_file)
  close(visium$matrix_file)
}

read_visium_data = function(data_dir) {
  positions_path = file.path(data_dir, 'tissue_positions_list.csv')
  barcodes_path = file.path(data_dir, 'barcodes.tsv.gz')
  features_path = file.path(data_dir, 'features.tsv.gz')
  matrix_path = file.path(data_dir, 'matrix.mtx.gz')
  if (!file.exists(positions_path)) write(paste("[ERROR]", positions_path, "does not"), stderr())
  if (!file.exists(barcodes_path)) write(paste("[ERROR]", barcodes_path, "does not"), stderr())
  if (!file.exists(features_path)) write(paste("[ERROR]", features_path, "does not"), stderr())
  if (!file.exists(matrix_path)) write(paste("[ERROR]", matrix_path, "does not"), stderr())
  
  positions = read.table(positions_path, sep=',', row.names=1)
  colnames(positions) = c('in_tissue', 'row', 'col', 'prow', 'pcol')
  
  barcodes = read.table(barcodes_path, sep='\t')[,1] # as vector
  features = read.table(features_path, sep='\t')[,1] # 1st col only
  read_cnts = read.table(matrix_path, sep=' ', skip=3,
                         col.names=c('feature', 'barcode', 'count'))

  data = list(
    positions = positions,
    barcodes = barcodes, 
    features = features, 
    counts = list()
  )
  
  for (fix in 1 : (length(features))) {
    feature = features[fix] # gene id -> gene symbol
    data$counts[[feature]] = read_cnts[read_cnts$feature==fix,]$count # save counts for gene
    names(data$counts[[feature]]) = read_cnts[read_cnts$feature==fix,]$barcode # rename rows as barcode ids
    names(data$counts[[feature]]) = barcodes[as.integer(names(data$counts[[feature]]))] # rename rows as barcodes
  }
  
  return(data)
}


# Create connectivity matrix -> C
create_connectivity_matrix = function(vdata) {
  positions_in_tissue = vdata$positions[vdata$positions$in_tissue==1,]
  barcodes_in_tissue = rownames(positions_in_tissue)
  nbarcodes_in_tissue = length(barcodes_in_tissue)
  
  C = matrix(0, nbarcodes_in_tissue, nbarcodes_in_tissue) # C: connectivity matrix
  rownames(C) = barcodes_in_tissue
  colnames(C) = barcodes_in_tissue
  
  for (barcode in barcodes_in_tissue) {
    conn_mat = list() # data holder for connectivity matrices
    row_i = positions_in_tissue[barcode, 'row']
    col_i = positions_in_tissue[barcode, 'col'] # vdata$counts[['A']][[barcode]] later
    # Set nearby nodes and check connectivity (only in_tissue)
    neighbors = subset(vdata$positions,
                       in_tissue==1 &
                         (
                           ((row==row_i-1) & (col==col_i-1)) |
                             ((row==row_i-1) & (col==col_i+1)) |
                             ((row==row_i) & (col==col_i-2)) |
                             ((row==row_i) & (col==col_i+2)) |
                             ((row==row_i+1) & (col==col_i-1)) |
                             ((row==row_i+1) & (col==col_i+1))
                         ))
    neighbor_barcodes = rownames(neighbors)
    C[barcode, neighbor_barcodes] = 1
  }
  
  W = C / rowSums(C) # W: weighted connectivity matrix
  
  conn_mat[['barcodes_in_tissue']] = barcodes_in_tissue
  conn_mat[['nbarcodes_in_tissue']] = nbarcodes_in_tissue
  conn_mat[['W']] = W
  conn_mat[['C']] = C
  
  return(conn_mat)
}

ape::Moran.I

RunMoransI(brain, tissue_positions_list)

UseMethod("RunMoransI")
methods(RunMoransI)

trace(RunMoransI, edit = T)
methods(RunMoransI) # Next step, ask for the method: 'princomp.default'

Seurat::RunMoransI
body(RunMoransI)
brain 
# <- brain_copy
brain_copy <- brain

# expmat <- as.matrix(GetAssayData(object = brain, slot = "counts"))
expr_matrix <- GetAssayData(brain, slot='counts', assay='SCT')

posit_mat <- brain@images$anterior1@coordinates[,c("col","row")]

brainMoran <- RunMoransI(expr_matrix, posit_mat, verbose = TRUE)

head(brainMoran)

head(brainMoran[sort(brainMoran$observed),])

brainMoran_ordered1 <- brainMoran[order(brainMoran$observed, decreasing= TRUE),]
brainMoran_ordered2 <- brainMoran[order(brainMoran$p.value, decreasing= FALSE),]
head(brainMoran_ordered1,5)
head(brainMoran_ordered2)
brainMoran_ordered1[grepl("Vxn", rownames(brainMoran_ordered1)),]

write.csv(brainMoran_ordered1, file = "../../../../rstudio/brainMoran.csv", row.names = TRUE)

FindSpatiallyVariableFeatures(
  brain,
  assay = 'SCT',
  slot = "scale.data",
  features = NULL,
  image = NULL,
  selection.method = c("markvariogram", "moransi"),
  r.metric = 5,
  x.cuts = NULL,
  y.cuts = NULL,
  nfeatures = 2000,
  verbose = TRUE,
)

brain <- FindSpatiallyVariableFeatures(brain, assay = 'SCT', features = 
                                         VariableFeatures(brain)[1:1000], selection.method = 'moransi')

# We would get top features from SpatiallyVariableFeatures 
top.features <- head(SpatiallyVariableFeatures(brain, selection.method = 'moransi'), 6)

SpatialFeaturePlot(brain, features = top.features, ncol = 3, alpha = c(0.1, 1))

## Calculate Moran's I
calculate_Morans_I_brute = function(feature, vdata, conn_mat) {
  # Row-wise (xi-xbar), (xi-xbar)^2 calculation
  X_values = vdata$counts[[feature]][conn_mat$barcodes_in_tissue]
  X = data.frame(value=X_values)
  Xmean = mean(X$value)
  X['sub_mean'] = X$value - Xmean
  X['sub_mean_sq'] = X$sub_mean^2
  
  # Row-wise xi*[x1...xn] calculation -> R
  Xmm = matrix(X$sub_mean) # X minus mean
  XmmtXmm = Xmm %*% t(Xmm) # 
  
  # Matrix-span W*R calculation
  WXmmtXmm = W * XmmtXmm
  
  # Moran's I = sum(W*R) / sum_i((xi-xbar)^2)
  moransI = sum(WXmmtXmm) / sum(X$sub_mean_sq)
  return(moransI)
}

# Calculate Moran's I using eq(5) from LeeS2001
calculate_Morans_I = function(feature, vdata, conn_mat) {
  # Simplify using: Xsm_i (smooth) = sum_j(w_ij * X_j)
  # Row-wise (xi-xbar), (xi-xbar)^2 calculation
  X_values = vdata$counts[[feature]][conn_mat$barcodes_in_tissue]
  X = data.frame(value=X_values)
  Xmean = mean(X$value)
  X['sub_mean'] = X$value - Xmean
  X['sub_mean_sq'] = X$sub_mean^2
  
  X['smooth'] = conn_mat$W %*% X[,'value'] # Xsm = W * X
  nom_ = (X$value - Xmean) * (X$smooth - Xmean) # nominator
  denom_ = X$sub_mean_sq # denominator; (X$value - Xmean)^2
  moransI = sum(nom_) / sum(denom_)
  return(moransI)
}

# Calculate R_X,Y of eq(16) from LeeS2001
calculate_r_sm = function(feature1, feature2, vdata, conn_mat) {
  # Set X, Y variables from two features
  X_values = vdata$counts[[feature1]][conn_mat$barcodes_in_tissue] # 1:X
  Y_values = vdata$counts[[feature2]][conn_mat$barcodes_in_tissue] # 2:Y
  X = data.frame(value=X_values)
  Xmean = mean(X$value)
  Y = data.frame(value=Y_values)
  Ymean = mean(Y$value)
  
  # Calculate R_X,Y - connectivity matrix shared between X and Y
  X['smooth'] = conn_mat$W %*% X[,'value'] # Xsm = W * X
  Y['smooth'] = conn_mat$W %*% Y[,'value'] # Ysm = W * Y
  Xmean_sm = mean(X$smooth) # muX
  Ymean_sm = mean(Y$smooth) # muY
  nom_XY_ = (X$smooth - Xmean_sm) * (Y$smooth - Ymean_sm) # nominator
  denom_X_ = X$smooth - Xmean_sm # denominator part for X
  denom_Y_ = Y$smooth - Ymean_sm # denominator part for Y
  r_sm = sum(nom_XY_) / sum(sqrt(sum(denom_X_^2)) * sqrt(sum(denom_Y_^2))) # eq(15)
  return(r_sm)
}


# Calculate L_X,Y of eq(16) from LeeS2001 - approximate; only works if feature1=feature2
calculate_L = function(feature1, feature2, vdata, conn_mat) {
  # Set X, Y variables from two features
  X_values = vdata$counts[[feature1]][conn_mat$barcodes_in_tissue] # 1:X
  Y_values = vdata$counts[[feature2]][conn_mat$barcodes_in_tissue] # 2:Y
  X = data.frame(value=X_values)
  Xmean = mean(X$value)
  Y = data.frame(value=Y_values)
  Ymean = mean(Y$value)
  
  #Calculate L_X,Y: per-gene spatial correlation
  X['smooth'] = conn_mat$W %*% X[,'value'] # Xsm = W * X
  Y['smooth'] = conn_mat$W %*% Y[,'value'] # Ysm = W * Y
  Xmean_sm = mean(X$smooth) # muX
  Ymean_sm = mean(Y$smooth) # muY
  nom_X_ = X$smooth - Xmean # nominator part for X
  nom_Y_ = Y$smooth - Ymean # nominator part for Y
  denom_X_ = X$value - Xmean # denominator part for X
  denom_Y_ = Y$value - Ymean # denominator part for Y
  L_XY = sqrt(sum(nom_X_^2)/sum(denom_X_^2)) * sqrt(sum(nom_Y_^2)/sum(denom_Y_^2))
  return(L_XY)
}

calculate_bsc = function(feature1, feature2, vdata, conn_mat) {
  # Set X, Y variables from two features
  X_values = vdata$counts[[feature1]][conn_mat$barcodes_in_tissue] # 1:X
  Y_values = vdata$counts[[feature2]][conn_mat$barcodes_in_tissue] # 2:Y
  X = data.frame(value=X_values)
  Xmean = mean(X$value)
  Y = data.frame(value=Y_values)
  Ymean = mean(Y$value)
  
  # Smoothened values
  X['smooth'] = conn_mat$W %*% X[,'value'] # Xsm = W * X
  Y['smooth'] = conn_mat$W %*% Y[,'value'] # Ysm = W * Y
  Xmean_sm = mean(X$smooth) # muX
  Ymean_sm = mean(Y$smooth) # muY

  # Calculate Peason's r(X,Y)
  r = (sum((X$value - Xmean) * (Y$value - Ymean))
       / (sqrt(sum((X$value - Xmean)^2)) * sqrt(sum((Y$value - Ymean)^2))))
  r_sm = (sum((X$smooth - Xmean_sm) * (Y$smooth - Ymean_sm))
          / (sqrt(sum((X$smooth - Xmean_sm)^2)) * sqrt(sum((Y$smooth - Ymean_sm)^2))))
  L_XX = (sum((X$smooth - Xmean)^2) / sum((X$value - Xmean)^2))
  L_YY = (sum((Y$smooth - Ymean)^2) / sum((Y$value - Ymean)^2))
  L_XY = sqrt(L_XX) * sqrt(L_YY) * r_sm
  
  # Group into Bivariate Spatial Correlation values list
  bsc = list('r' = r, 'r_sm' = r_sm,
             'L_XX' = L_XX, 'L_YY' = L_YY, 'L_XY' = L_XY)
  
  return(bsc)
}



############
##  MAIN  ##
############

if (!interactive()) { # don't run if sourced
  # Read test data fig1 graph A/B/C 
  setwd('~/BSC')
  test_position_data = GeneTestData('A')
  
  spatial = Spatial("visium/positions_list.csv")
  write_positions_data(test_position_data, spatial)
  
  visium = Visium()
  gsyms = c('A', 'B', 'C')
  write_visium_data(gsyms, visium)
  
  # Read Visium-style data
  vdata = read_visium_data(data_dir='./visium')
  
  # Create connectivity matrix C(raw) and W(weighted)
  conn_mat = create_connectivity_matrix(vdata)
  
  # Calculate Moran's I using eq(5) from LeeS2001
  feature = 'A'
  moransI = calculate_Morans_I(feature, vdata, conn_mat)
  
  # Calculate statistics from LeeS (2001) Table 1
  features = c('A', 'B', 'C')
  for (feature1 in features) {
    for (feature2 in features) {
      bsc = calculate_bsc(feature1, feature2, vdata, conn_mat) # r_XY(smooth)
      
      print_line = sprintf("[%s-%s] %.3f %.3f %.3f %.3f %.3f", 
                           feature1, feature2, 
                           bsc$L_XX, bsc$L_YY, bsc$r_sm, bsc$r, bsc$L_XY)
      print(print_line)
    }
  }
}

feature1 = "Gng4"
feature2 = "Pcbp3"
bsc = calc_bsc(feature1, feature2, brain, conn_mat) # r_XY(smooth)

print_line = sprintf("[%s-%s] %.3f %.3f %.3f %.3f %.3f", 
                     feature1, feature2, 
                     bsc$L_XX, bsc$L_YY, bsc$r_sm, bsc$r, bsc$L_XY)
print(print_line)

feature1 = "Vxn"
feature2 = "Pcbp3"
bsc = calc_bsc(feature1, feature2, brain, conn_mat, assay='SCT')
print_line = sprintf("[%s-%s] %.3f %.3f %.3f %.3f %.3f", 
                     feature1, feature2,
                     bsc$L_XX, bsc$L_YY, bsc$r_sm, bsc$r, bsc$L_XY)
print(print_line)

feature1 = "Vxn"
feature2 = "Olfm1"

SpatialFeaturePlot(brain, features = c(feature1, feature2), ncol = 3, 
                   alpha = c(0.1, 1), max.cutoff = 5,)