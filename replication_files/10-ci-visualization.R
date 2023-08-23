
## prop <- .125
## xs <- seq(0, 1, .001)
## ys <- sapply(xs, function(x) KL(p = c(x, 1 - x), q = c(prop, 1 - prop)))

## K <- 8
## N <- 1000
## alpha <- .05

## thresh <- 1 / N * log(2 * K / alpha)

################################################################
## test multinomial confidence sets for this graph            ##
##      U                                                     ##
##     / \                                                    ##
##    v   v                                                   ##
##   X --> Y                                                  ##
## using Malloy, Tripathy, and Nowak (2021)                   ##
## "Optimal Confidence Regions for the Multinomial Parameter" ##
################################################################



##########################
## packages & functions ##
##########################

## math
library(gtools)
library(MASS)
library(binom)
library(geometry)
## library(mvmesh)  # generate grid on simplex

## data manipulation
library(data.table)

## parallel
library(foreach)
library(doParallel)
registerDoParallel(cores = detectCores() - 1)

## graphics
library(ggplot2)
library(rgl)
## library(car)

if (!file.exists('results/ci_images')){ 
    dir.create('results/ci_images')
}

`%.%` <- paste0

grepv <- function(...){
  args <- list(...)
  args$value <- TRUE
  do.call(grep, args)
}

KL <- function(p, q){
  if (length(p) != length(q)){
    stop('distributions must have same dimension')
  }
  if (!all.equal(sum(p), 1)){
    stop('distributions must sum to 1')
  }
  if (!all.equal(sum(q), 1)){
    stop('distributions must sum to 1')
  }
  sum(p * log(p / q))
}

KL.vectorized <- function(p, qs){
  if (length(p) != ncol(qs)){
    stop('distributions must have same dimension')
  }
  if (!all.equal(sum(p), 1)){
    stop('distributions must sum to 1')
  }
  if (!all.equal(rowSums(qs), rep(1, nrow(qs)))){
    stop('distributions must sum to 1')
  }
  colSums(p * log(p / t(qs)))  # exploit auto-recycling over columns
}

darkred <- '#850617'
lightred <- '#A51C2F'
blue <- '#4C6C9C'



####################################
## set up data-generating process ##
####################################

## each row is a possible response function: r_X()
## each col is a possible input (only one):     ()
## each cell is an output:                   r_X()
rX <- as.matrix(data.table(X = 0:1))

## each row is a possible response function: r_Y(.)
## each col is a possible input:                (x)
## each cell is an output:                   r_Y(x)
rY <- as.matrix(
  expand.grid(
    Y.x0 = 0:1,
    Y.x1 = 0:1
    )
)

## randomly generate a joint population distribution
##   over response variables for X and Y
pop <- data.table(
  expand.grid(rX.ind = 1:nrow(rX),
              rY.ind = 1:nrow(rY)
              )
)
pop <- cbind(pop, pop[, .(X = rX[rX.ind,], rY[rY.ind,])])
set.seed(02139)
pop[, p := as.numeric(rdirichlet(n = 1, alpha = rep(1, .N)))]

## look up realized variables from potential outcome table. notes:
##   1. define 'po', a tmpvar for the potential outcome that will be
##      looked up for each row, e.g. look up Y(0) for row i and Y(1) for row i'
##   2. uses array indices to pick out cells in matrix M, i.e.
##      M[cbind(rows, cols)] = c(M[rows[1], cols[1]], M[rows[2], cols[2]], ...)
pop[, po := 'X']          # for this variable we simply look up value of X
pop[, X := rX[cbind(rX.ind, chmatch(po, colnames(rX)))]]
pop[, po := 'Y.x' %.% X]  # X->Y, so look up Y(0) if X=0, look up Y(1) if X=1
pop[, Y := rY[cbind(rY.ind, chmatch(po, colnames(rY)))]]

pop.margins <- pop[, .(p = sum(p)), by = .(X, Y)]
pop.margins[, name := sprintf('x%s.y%s', X, Y)]
setkey(pop.margins, name)



#################
## draw sample ##
#################

N <- 1e3
set.seed(02139)
sample <- pop[sample(1:.N, size = N, replace = TRUE, prob = p),
              .(X, Y,            # observed
                Y.x0, Y.x1,      # unobserved
                rX.ind, rY.ind
                )
              ]

sample.summary <- sample[, .(prop = .N / N), by = .(X, Y)]
sample.summary[, name := sprintf('x%s.y%s', X, Y)]
setkey(sample.summary, name)



########################################
## construct polytope confidence sets ##
########################################

## for each unique (x, y), let
##   phat.xy = 1/N sum_i 1{ X_i = x, Y_i = y }  (observed)
##   and p.xy = Pr(X_i = x, Y_i = y)            (unobserved)
##
## then we have from equation 11 of Malloy, Tripathy, and Nowak (2021)
##   KL(
##       (phat.xy, 1 - phat.xy),
##       (p.xy   , 1 - p.xy   )
##      )  <=  log(2 * K / delta) / N
## note: this requires K ≤ e * ( N / 8pi )^(1/3)
##
## which implies the system of equations
##   { cilo.xy <= p.xy <= cihi.xy : x, y }
## is satisfied with probability 1 - delta
##
## for each (x, y) bucket we will solve numerically for cilo.xy
##   KL(
##       (phat.xy, 1 - phat.xy),
##       (cilo.xy, 1 - cilo.xy)
##      )  ==  log(2 * K / delta) / N
##   subject to cilo.xy < phat.xy, and similarly
##   KL(
##       (phat.xy, 1 - phat.xy),
##       (cihi.xy, 1 - cihi.xy)
##      )  ==  log(2 * K / delta) / N
##   subject to cihi.xy > phat.xy

get.confset.polytope <- function(sample.summary, N, alpha = .95){

  delta <- 1 - alpha
  K <- nrow(sample.summary)  # number of unique buckets for (X, Y)
  thresh <- log(2 * K / delta) / N

  confset <- matrix(NA_real_,
                             nrow = K,
                             ncol = 2,
                             dimnames = list(name = sprintf('x%s.y%s',
                                                            sample.summary$X,
                                                            sample.summary$Y
                                                            ),
                                             ci = c('cilo', 'cihi')
                                             )
                             )
  confset <- data.table(confset, name = rownames(confset))

  for (k in 1:K){

    cilo.opt <- optimize(
      f = function(cilo.star){
        (
          KL(p = c(sample.summary$prop[k], 1 - sample.summary$prop[k]),
             q = c(cilo.star, 1 - cilo.star)
             ) - thresh
        )^2
      },
      lower = 0,
      upper = sample.summary$prop[k],
      maximum = FALSE
    )$minimum

    cihi.opt <- optimize(
      f = function(cihi.star){
        (
          KL(p = c(sample.summary$prop[k], 1 - sample.summary$prop[k]),
             q = c(cihi.star, 1 - cihi.star)
             ) - thresh
        )^2
      },
      lower = sample.summary$prop[k],
      upper = 1,
      maximum = FALSE
    )$minimum

    confset[k, cilo := cilo.opt]
    confset[k, cihi := cihi.opt]

  }

  return(confset)

}

## construct polytope confidence set for a single sample
confset <- get.confset.polytope(sample.summary, N = N)

## reported stats in paper
pop[, sum(p), by = .(X, Y)]
sample.summary
confset[, sprintf('%s: [%0.3f, %0.3f]', name, cilo, cihi)]




## check if confidence set contains truth
confset <- merge(confset, pop.margins, by = 'name')
confset[, all(between(p, cilo, cihi))]



alpha <- .95
delta <- 1 - alpha
K <- nrow(sample.summary)  # number of unique buckets for (X, Y)
grid.1d <- seq(0, 1, .01)

## empirical distribution
p.hat <- sample.summary$prop

grid <- lapply(
  1:K,
  function(k){
    ## each col will be a candidate distr that perturbs one bucket's size
    ##   (starting from the empirical distr)
    p.stars.1d <- replicate(n = length(grid.1d), p.hat)
    ## modify candidate distrs by changing target bucket in fixed steps
    p.stars.1d[k,] <- grid.1d
    ## rescale all other buckets to sum to 1
    p.stars.1d[-k,] <- sweep(p.stars.1d[-k,],
                             MARGIN = 2,
                             STATS = (1 - grid.1d) / colSums(p.stars.1d[-k,]),
                             FUN = `*`
                             )
    ## compute upper bound for p-values
    pvals.upperbound <- apply(
      p.stars.1d,
      MARGIN = 2,
      function(p.star){
        (N + 1)^(2 * K) * exp(-N * KL(p.hat, p.star) )
      })
    ## keep all values that fail to reject
    return(grid.1d[pvals.upperbound >= delta])
  })
p.stars <- do.call(CJ, grid)
p.stars <- as.matrix(p.stars)
p.stars <- p.stars[rowSums(p.stars) == 1,]
colnames(p.stars) <- sample.summary$name

plot.base <- function(bbox = NULL, line.scale = 1, cex.scale = 1){
  ## viewer angle
  open3d(
    ## userMatrix = rbind(c( 0.8458892, -0.5333573, 0.001048209, 0),
    ##                    c( 0.1849379,  0.2951484, 0.937381983, 0),
    ##                    c(-0.5002691, -0.7927276, 0.348300904, 0),
    ##                    c( 0.0000000,  0.0000000, 0.000000000, 1)
    ##                    ),
    userMatrix = rbind(c( 0.5836757, -0.8115045, 0.02797217, 0),
                       c( 0.3164214,  0.2590422, 0.91256440, 0),
                       c(-0.7477965, -0.5237909, 0.40797415, 0),
                       c( 0.0000000,  0.0000000, 0.00000000, 1)
                       ),
    windowRect = c(73, 105, 1921, 1118)
  )
  ## expand plot outwards for visibility using invisible points
  if (is.null(bbox)){
    bbox <- expand.grid(apply(p.stars, 2, range)[, 1] + c(-.06, .06),
                        apply(p.stars, 2, range)[, 2] + c(-.06, .06),
                        apply(p.stars, 2, range)[, 3] + c(-.06, .06)
                        )
  }
  points3d(x = bbox[, 1],
           y = bbox[, 2],
           z = bbox[, 3],
           col = NA,
           alpha = 0
           )
  ## axes & labels
  axes3d(col = 'black', nticks = 3, cex = cex.scale)
  mtext3d(text = 'Pr(X=0, Y=0)',
          edge = 'x++',
          line = 2.5 * line.scale,
          at = mean(range(bbox[,1])),
          cex = cex.scale
          )
  mtext3d(text = 'Pr(X=0, Y=1)',
          edge = 'y--',
          line = 2 * line.scale,
          at = c(.25, .75) %*% range(bbox[,2]),
          cex = cex.scale
          )
  mtext3d(text = 'Pr(X=1, Y=0)',
          edge = 'z-+',
          line = 3 * line.scale,
          at = c(.5, .5) %*% range(bbox[,3]),
          cex = cex.scale
          )
}



###########################################
## visualize one-vs-rest confidence sets ##
###########################################

## for each unique (x, y), let
##   phat.zxy = 1/N sum_i 1{ Z_i = z, X_i = x, Y_i = y }  (observed)
##   and p.xy = Pr(Z_i = z, X_i = x, Y_i = y)             (unobserved)
##
## then we have from equation 11 of Malloy, Tripathy, and Nowak (2021)
##   KL(
##       (phat.zxy, 1 - phat.zxy),
##       (p.zxy   , 1 - p.zxy   )
##      )  <=  log(2 * K / delta) / N
## note: this requires K ≤ e * ( N / 8pi )^(1/3)
##
## which implies the system of equations
##   { cilo.zxy <= p.zxy <= cihi.zxy : z, x, y }
## is satisfied with probability alpha = 1 - delta
##
## for each (z, x, y) bucket we will solve numerically for cilo.zxy
##   KL(
##       (phat.zxy, 1 - phat.zxy),
##       (cilo.zxy, 1 - cilo.zxy)
##      )  ==  log(2 * K / delta) / N
##   subject to cilo.zxy < phat.zxy, and similarly
##   KL(
##       (phat.zxy, 1 - phat.zxy),
##       (cihi.zxy, 1 - cihi.zxy)
##      )  ==  log(2 * K / delta) / N
##   subject to cihi.zxy > phat.zxy

get.confset.polytope <- function(sample.summary, alpha = .95){
  ## constants
  delta <- 1 - alpha
  K <- nrow(sample.summary)  # number of unique buckets for (X, Y)
  thresh <- log(2 * K / delta) / N
  ## initialize empty matrix of confidence region boundaries
  ## (upper and lower boundary along each dimension)
  confset <- matrix(NA_real_,
                    nrow = K,
                    ncol = 2,
                    dimnames = list(name = sprintf('x%s.y%s',
                                                   sample.summary$X,
                                                   sample.summary$Y
                                                   ),
                                    ci = c('cilo', 'cihi')
                                    )
                    )
  confset <- data.table(confset, name = rownames(confset))
  ## take each dimension and split it in half
  ## then numerically optimize to find boundaries of eq 11
  for (k in 1:K){
    if (sample.summary$prop[k] == 0){  # unless point is already on boundary
      cilo.opt <- 0
    } else {
      cilo.opt <- optimize(
        f = function(cilo.star){
          (
            KL(p = c(sample.summary$prop[k], 1 - sample.summary$prop[k]),
               q = c(cilo.star, 1 - cilo.star)
               ) - thresh
          )^2
        },
        lower = 0,
        upper = sample.summary$prop[k],
        maximum = FALSE
      )$minimum
    }
    if (sample.summary$prop[k] == 1){  # unless point is already on boundary
      cihi.opt <- 1
    } else {
      cihi.opt <- optimize(
        f = function(cihi.star){
          (
            KL(p = c(sample.summary$prop[k], 1 - sample.summary$prop[k]),
               q = c(cihi.star, 1 - cihi.star)
               ) - thresh
          )^2
        },
        lower = sample.summary$prop[k],
        upper = 1,
        maximum = FALSE
      )$minimum
    }
    confset[k, cilo := cilo.opt]
    confset[k, cihi := cihi.opt]
  }
  return(confset)
}

get.confset.polytope.inequalities <- function(confset){
  D <- diag(rep(1, nrow(confset)))
  colnames(D) <- confset$name
  return(
    rbind(
      data.table(D,
                 dir = '>=',
                 rhs = confset$cilo
                 ),
      data.table(D,
                 dir = '<=',
                 rhs = confset$cihi
                 )
    )
  )
}

confset <- get.confset.polytope(sample.summary)

plot.base()

points3d(x = confset[name == 'x0.y0', cilo],
         y = confset[name == 'x0.y1', cilo],
         z = confset[name == 'x1.y0', cilo]
         )
points3d(x = confset[name == 'x0.y0', cilo],
         y = confset[name == 'x0.y1', cilo],
         z = confset[name == 'x1.y0', cihi]
         )
points3d(x = confset[name == 'x0.y0', cilo],
         y = confset[name == 'x0.y1', cihi],
         z = confset[name == 'x1.y0', cilo]
         )

confset <- as.matrix(confset[name != 'x1.y1', .(cilo, cihi)],
                     rownames = confset[name != 'x1.y1', name]
                     )

confset.points <- expand.grid(x0.y0.side = c('cilo', 'cihi'),
                              x0.y1.side = c('cilo', 'cihi'),
                              x1.y0.side = c('cilo', 'cihi'),
                              stringsAsFactors = FALSE
                              )
confset.points <- data.table(confset.points)
confset.points[, x0.y0 := confset[cbind('x0.y0', x0.y0.side)]]
confset.points[, x0.y1 := confset[cbind('x0.y1', x0.y1.side)]]
confset.points[, x1.y0 := confset[cbind('x1.y0', x1.y0.side)]]

plot.base(bbox = expand.grid(x = confset[1,] + c(-.035, .035),
                             y = confset[2,] + c(-.035, .035),
                             z = confset[3,] + c(-.035, .035)
                             ),
          cex.scale = 2,
          line.scale = 1.5
          )
rgl.triangles(x = confset.points$x0.y0[as.numeric(t(combinations(8, 3)))],
              y = confset.points$x0.y1[as.numeric(t(combinations(8, 3)))],
              z = confset.points$x1.y0[as.numeric(t(combinations(8, 3)))],
              col = lightred,
              alpha = 1
              )
rgl.snapshot('./results/ci_images/ci_box.png', fmt = 'png')
rgl.close()



##########################################
## visualize asymptotic confidence sets ##
##########################################

confset.mu <- sample.summary$prop[-K]
names(confset.mu) <- sample.summary$name[-K]
confset.Sigma <- (diag(confset.mu) - outer(confset.mu, confset.mu)) / N

plot.base(bbox = expand.grid(x = confset[1,] + c(-.035, .035),
                             y = confset[2,] + c(-.035, .035),
                             z = confset[3,] + c(-.035, .035)
                             ),
          cex.scale = 2,
          line.scale = 1.5
          )
plot3d(ellipse3d(confset.Sigma, centre = confset.mu),
       color = lightred,
       add = TRUE
       )
rgl.snapshot('./results/ci_images/ci_ellipse.png', fmt = 'png')
rgl.close()
