library(causaloptim)
library(data.table)
library(gtools)
library(ggplot2)
library(xtable)

`%.%` <- paste0



###########
## style ##
###########

red <- '#A51C30'
lightred <- '#E16273'
blue <- '#4E84C4'
darkblue <- '#0C2C5C'
lightblue <- '#4C6C9C'



############################
## read bounds trajectory ##
############################

read.traj <- function(path, tol = .001, extend = 0){

  traj <- fread(path)
  traj[, less.than.tol := abs(primal - dual) <= tol]
  traj[is.na(less.than.tol), less.than.tol := FALSE]
  traj[, tol.keep := cumsum(less.than.tol) == 1, by = bound]
  traj <- traj[is.na(primal) | abs(primal - dual) >= tol | tol.keep,]
  traj[dual <= -1, dual := -1]
  traj[dual >= 1, dual := 1]

  lb <- traj[bound == 'lb',]
  ub <- traj[bound == 'ub',]

  ## for f(t) when t not in list, take f(t') for max(t' : t' < t)
  lb.prim.fun <- approxfun(lb$time,
                           lb$primal,
                           method = 'constant',
                           f = 0,
                           rule = 2
                           )
  lb.dual.fun <- approxfun(c(0, lb$time),
                           c(-1, lb$dual),
                           method = 'constant',
                           f = 0,
                           rule = 2
                           )
  ub.prim.fun <- approxfun(ub$time,
                           ub$primal,
                           method = 'constant',
                           f = 0,
                           rule = 2
                           )
  ub.dual.fun <- approxfun(c(0, ub$time),
                           c(1, ub$dual),
                           method = 'constant',
                           f = 0,
                           rule = 2
                           )

  unique.times <- sort(unique(c(0, lb$time, ub$time)))
  if (extend > 0){
    unique.times <- c(unique.times, max(unique.times) * (1 + extend))
  }
  times <- seq(min(unique.times),
               max(unique.times),
               diff(range(unique.times)) / 1000
               )

  traj <- data.table(seconds = times,
                     lb.prim = lb.prim.fun(times),
                     lb.dual = lb.dual.fun(times),
                     ub.prim = ub.prim.fun(times),
                     ub.dual = ub.dual.fun(times)
                     )
  setkey(traj, seconds)
  return(traj)

}

clean.scip.log <- function(log.path){
  ## read in
  scip.log <- readLines(log.path)
  ## drop blank lines
  scip.log <- scip.log[scip.log != '']
  ## drop intro text
  scip.log <- scip.log[-(1:max(grep('^\\*+$', scip.log)))]
  ## drop intermediate headers
  scip.log <- scip.log[
    scip.log != " time | node  | left  |LP iter|LP it/n|mem/heur|mdpt |vars |cons |rows |cuts |sepa|confs|strbr|  dualbound   | primalbound  |  gap   | compl. "
  ]
  ## reappend top header
  scip.log <- c(" time | node  | left  |LP iter|LP it/n|mem/heur|mdpt |vars |cons |rows |cuts |sepa|confs|strbr|  dualbound   | primalbound  |  gap   | compl. ",
               scip.log
               )
  ## drop outro text
  scip.log <- scip.log[1:(grep('SCIP Status', scip.log) - 1)]
  ## convert to comma separated
  scip.log <- gsub('|', ',', scip.log, fixed = TRUE)
  ## write out
  writeLines(scip.log,
             gsub('\\.log', '_clean.log', log.path)
             )
}

read.traj.scip <- function(lb.path, ub.path, tol = .001, max.time = Inf, extend = 0){

  lb <- fread(lb.path)
  lb[, primalbound := as.numeric(primalbound)]
  lb[, dualbound := as.numeric(dualbound)]
  lb[, less.than.tol := abs(primalbound - dualbound) <= tol]
  lb[is.na(less.than.tol), less.than.tol := FALSE]
  lb[, tol.keep := cumsum(less.than.tol) == 1]
  lb <- lb[is.na(primalbound) | abs(primalbound - dualbound) >= tol | tol.keep,]
  lb[dualbound <= -1, dualbound := -1]
  ub <- fread(ub.path)
  ub[, primalbound := as.numeric(primalbound)]
  ub[, dualbound := as.numeric(dualbound)]
  ub[, less.than.tol := abs(primalbound - dualbound) <= tol]
  ub[is.na(less.than.tol), less.than.tol := FALSE]
  ub[, tol.keep := cumsum(less.than.tol) == 1]
  ub <- ub[is.na(primalbound) | abs(primalbound - dualbound) >= tol | tol.keep,]
  ub[dualbound >= 1, dualbound := 1]

  lb[, time := gsub('[*L]', '', time)]
  ub[, time := gsub('[*L]', '', time)]

  lb[, time.seconds := grepl('s', time)]
  lb[, time.minutes := grepl('m', time)]
  ub[, time.seconds := grepl('s', time)]
  ub[, time.minutes := grepl('m', time)]

  lb[, time := as.numeric(gsub('[ sm]', '', time))]
  ub[, time := as.numeric(gsub('[ sm]', '', time))]

  lb[time.minutes == TRUE, time := time * 60]
  ub[time.minutes == TRUE, time := time * 60]

  lb <- lb[time <= max.time,]
  ub <- ub[time <= max.time,]

  setkey(lb, time)
  setkey(ub, time)

  lb[, time.to.next := c(diff(time), NA)]
  ub[, time.to.next := c(diff(time), NA)]
  min.time <- unique(na.omit(c(lb[, time.to.next], ub[, time.to.next])))
  min.time <- min(min.time[min.time > 0])
  if (min.time == Inf){
    min.time <- .1
  }
  lb[is.na(time.to.next), time.to.next := min.time]
  ub[is.na(time.to.next), time.to.next := min.time]

  lb[, time.duplicate := .N, by = 'time']
  ub[, time.duplicate := .N, by = 'time']

  ## modifly duplicated timestamps to spread uniformly over period until next timestamp
  lb[time.duplicate > 1,
     time := time + (0:(.N-1)) / .N * max(time.to.next),
     by = 'time'
     ]
  ub[time.duplicate > 1,
     time := time + (0:(.N-1)) / .N * max(time.to.next),
     by = 'time'
     ]

  ## for f(t) when t not in list, take f(t') for max(t' : t' < t)
  lb.prim.fun <- approxfun(lb$time,
                           lb$primalbound,
                           method = 'constant',
                           f = 0,
                           rule = 2
                           )
  lb.dual.fun <- approxfun(lb$time,
                           lb$dualbound,
                           method = 'constant',
                           f = 0,
                           rule = 2
                           )
  ub.prim.fun <- approxfun(ub$time,
                           ub$primalbound,
                           method = 'constant',
                           f = 0,
                           rule = 2
                           )
  ub.dual.fun <- approxfun(ub$time,
                           ub$dualbound,
                           method = 'constant',
                           f = 0,
                           rule = 2
                           )

  unique.times <- sort(unique(c(lb$time, ub$time)))
  if (extend > 0){
    unique.times <- c(unique.times, max(unique.times) * (1 + extend))
  }
  times <- seq(min(unique.times),
               max(unique.times),
               diff(range(unique.times)) / 1000
               )

  traj <- data.table(seconds = times,
                     lb.prim = lb.prim.fun(times),
                     lb.dual = lb.dual.fun(times),
                     ub.prim = ub.prim.fun(times),
                     ub.dual = ub.dual.fun(times)
                     )
  traj <- traj[seconds > 0,]
  return(traj)

}

plot.bounds <- function(traj,
                        lb.theory = NA,
                        ub.theory = NA,
                        xlab = 'Seconds',
                        ylab = 'ATE',
                        size = 36,
                        family = NULL
                        ){
  p <- ggplot(traj) +
      geom_ribbon(aes(x = seconds,
                      ymin = lb.dual,
                      ymax = ub.dual
                      ),
                  color = NA,
                  fill = lightred
                  ) +
      geom_line(aes(x = seconds,
                    y = ub.dual
                    ),
                color = red,
                lineend = 'square',
                size = 3
                ) +
      geom_line(aes(x = seconds,
                    y = lb.dual
                    ),
                color = red,
                lineend = 'square',
                size = 3
                ) +
      geom_ribbon(aes(x = seconds,
                      ymin = lb.prim,
                      ymax = ub.prim
                      ),
                  fill = lightblue,
                  ) +
      geom_line(aes(x = seconds,
                    y = ub.prim
                    ),
                color = darkblue,
                lineend = 'square',
                size = 1
                ) +
      geom_line(aes(x = seconds,
                    y = lb.prim
                    ),
                color = darkblue,
                lineend = 'square',
                size = 1
                )
    if (!is.na(lb.theory) & !is.na(ub.theory)){
      p <- p +
        geom_errorbar(aes(x = seconds,
                          ymin = lb,
                          ymax = ub
                          ),
                      data = data.frame(seconds = 1.1 * max(traj$seconds),
                                        lb = lb.theory,
                                        ub = ub.theory
                                        ),
                      width = .05 * max(traj$seconds),
                      size = 2
                      )
    }
  p <- p +
    xlab(xlab) +
    ylab(ylab) +
    geom_hline(yintercept = 0, linetype = 'ff') +
    theme_classic(base_size = size) +
    theme(axis.text = theme_light()$axis.text,
          axis.line = element_line(color = 'gray70',
                                   lineend = 'square',
                                   size = 1.5
                                   ),
          axis.ticks = element_line(color = 'gray70',
                                    size = 1.5
                                    ),
          text = element_text(family = family)
          ) +
    xlim(0, 1.15 * max(traj$seconds))
  return(p)
}



###################
## noncompliance ##
###################

set.seed(02139)

## ## version 1: check with randomly generated dgp ###

## ## unconfounded Z: define joint distribution over X & Y response variables
## R.Z <- data.table(Z = 0:1)
## R.Z[, p.Z := as.numeric(rdirichlet(1, alpha = rep(1, .N)))]

## ## confounded X -> Y: define joint distribution over X & Y response variables
## R.XY <- CJ(X0 = 0:1,
##            X1 = 0:1,
##            Y0 = 0:1,
##            Y1 = 0:1
##            )
## R.XY[, p.XY := as.numeric(rdirichlet(1, alpha = rep(1, .N)))]

## ## combine indep margins into joint distr over X, Ya, Yb response variables
## ind.ZXY <- CJ(ind.Z = 1:nrow(R.Z),
##               ind.XY = 1:nrow(R.XY)
##               )
## R.ZXY <- cbind(R.Z[ind.ZXY$ind.Z,],
##                R.XY[ind.ZXY$ind.XY]
##                )
## ## generate distribution by combining independent margins
## R.ZXY[, p := p.Z * p.XY]
## R.ZXY[, p.Z := NULL]
## R.ZXY[, p.XY := NULL]

## ## look up factual outcome from potential outcomes
## R.ZXY[, X := ifelse(Z == 0, yes = X0, no = X1)]
## R.ZXY[, Y := ifelse(X == 0, yes = Y0, no = Y1)]

## ## observed quantities
## obs <- R.ZXY[, .(p = sum(p)), by = c('Z', 'X', 'Y')]



### version 2: check with autobounds ###

obs <- fread('data/iv.csv')
obs[, p := prob]
obs[, prob := NULL]

## p( x, y | z=0 )
p.z0 <- obs[Z == 0, .(X, Y, p = p / sum(p))]
p.z1 <- obs[Z == 1, .(X, Y, p = p / sum(p))]

### causaloptim results on ate, using "justified assumptions" (fig 6b) ###

G <- graph_from_literal(Ul -+ Z -+ X -+ Y, Ur -+ X, Ur -+ Y)
V(G)$leftside <- c(1, 1, 0, 0, 0)
V(G)$latent <- c(1, 0, 0, 0, 1)
V(G)$nvals <- c(Inf, 2, 2, 2, Inf)
E(G)$rlconnect <- c(0, 0, 0, 0, 0)
E(G)$edge.monotone <- c(0, 0, 0, 0, 0)

problem <- analyze_graph(G,
                         constraints = NULL,
                         effectt = 'p{Y(X = 1) = 1} - p{Y(X = 0) = 1}'
                         )

## pab_c: P(X = a, Y = b | Z = c)
bounds <- optimize_effect(problem)
bounds

## ## RESULT:
## MAX {
## p00_0 - p00_1 + p10_0 - 2 p10_1 - 2 p01_1
## - p00_0 + p00_1 - p10_0 - p01_0
## - p00_0 + p00_1 - 2 p10_0 + p10_1 - 2 p01_0
## p00_0 - p00_1 - p10_1 - p01_1
## - p10_0 - p01_0
## - p10_1 - p01_1
## p00_0 - p00_1 - p10_0 - p10_1 - p01_0
## - p00_0 + p00_1 - p10_0 - p10_1 - p01_1
## }
## MIN {
## - p00_0 - p10_0 - p10_1 - 2 p01_1 + 2
## - p00_1 - p10_0 - p10_1 - 2 p01_0 + 2
## - p10_0 - p01_1 + 1
## p00_1 - 2 p10_0 + p10_1 - p01_0 + 1
## - p10_0 - p01_0 + 1
## - p10_1 - p01_0 + 1
## - p10_1 - p01_1 + 1
## p00_0 + p10_0 - 2 p10_1 - p01_1 + 1
## }

## the following is generated using emacs query-replace-regex:
## FROM: P(X = \([01]\), Y = \([01]\) | Z = \([01]\))
## TO:   p.z\3[X == \1 & Y == \2, p]
lb.causaloptim <- max(
  p.z0[X == 0 & Y == 0, p] - p.z1[X == 0 & Y == 0, p] + p.z0[X == 1 & Y == 0, p] - 2*p.z1[X == 1 & Y == 0, p] - 2*p.z1[X == 0 & Y == 1, p],
  - p.z0[X == 0 & Y == 0, p] + p.z1[X == 0 & Y == 0, p] - p.z0[X == 1 & Y == 0, p] - p.z0[X == 0 & Y == 1, p],
  - p.z0[X == 0 & Y == 0, p] + p.z1[X == 0 & Y == 0, p] - 2*p.z0[X == 1 & Y == 0, p] + p.z1[X == 1 & Y == 0, p] - 2*p.z0[X == 0 & Y == 1, p],
  p.z0[X == 0 & Y == 0, p] - p.z1[X == 0 & Y == 0, p] - p.z1[X == 1 & Y == 0, p] - p.z1[X == 0 & Y == 1, p],
  - p.z0[X == 1 & Y == 0, p] - p.z0[X == 0 & Y == 1, p],
  - p.z1[X == 1 & Y == 0, p] - p.z1[X == 0 & Y == 1, p],
  p.z0[X == 0 & Y == 0, p] - p.z1[X == 0 & Y == 0, p] - p.z0[X == 1 & Y == 0, p] - p.z1[X == 1 & Y == 0, p] - p.z0[X == 0 & Y == 1, p],
  - p.z0[X == 0 & Y == 0, p] + p.z1[X == 0 & Y == 0, p] - p.z0[X == 1 & Y == 0, p] - p.z1[X == 1 & Y == 0, p] - p.z1[X == 0 & Y == 1, p]
)
ub.causaloptim <- min(
  - p.z0[X == 0 & Y == 0, p] - p.z0[X == 1 & Y == 0, p] - p.z1[X == 1 & Y == 0, p] - 2*p.z1[X == 0 & Y == 1, p] + 2,
  - p.z1[X == 0 & Y == 0, p] - p.z0[X == 1 & Y == 0, p] - p.z1[X == 1 & Y == 0, p] - 2*p.z0[X == 0 & Y == 1, p] + 2,
  - p.z0[X == 1 & Y == 0, p] - p.z1[X == 0 & Y == 1, p] + 1,
  p.z1[X == 0 & Y == 0, p] - 2*p.z0[X == 1 & Y == 0, p] + p.z1[X == 1 & Y == 0, p] - p.z0[X == 0 & Y == 1, p] + 1,
  - p.z0[X == 1 & Y == 0, p] - p.z0[X == 0 & Y == 1, p] + 1,
  - p.z1[X == 1 & Y == 0, p] - p.z0[X == 0 & Y == 1, p] + 1,
  - p.z1[X == 1 & Y == 0, p] - p.z1[X == 0 & Y == 1, p] + 1,
  p.z0[X == 0 & Y == 0, p] + p.z0[X == 1 & Y == 0, p] - 2*p.z1[X == 1 & Y == 0, p] - p.z1[X == 0 & Y == 1, p] + 1
)



### autobounds results on ate, using "justified assumptions" (fig 6b) ###

traj <- read.traj('results/iv_ate_just.csv')
lb.autobounds <- traj[, tail(lb.dual, 1)]
round(lb.autobounds, 2)
ub.autobounds <- traj[, tail(ub.dual, 1)]
round(ub.autobounds, 2)

## same result as causaloptim, to tolerance of 1e-5

lb.causaloptim
ub.causaloptim

lb.autobounds
ub.autobounds



### autobounds results for ate and late, using cautious/justified/erroneous ###

traj <- read.traj('results/iv_ate_cautious.csv')
sprintf('ate bounds with overly cautious assumptions: [%0.2f, %0.2f]',
        traj[, tail(lb.dual, 1)],
        traj[, tail(ub.dual, 1)]
        )

traj <- read.traj('results/iv_ate_just.csv')
sprintf('ate bounds with justified assumptions: [%0.2f, %0.2f]',
        traj[, tail(lb.dual, 1)],
        traj[, tail(ub.dual, 1)]
        )

traj <- read.traj('results/iv_late_cautious.csv')
sprintf('late bounds with overly cautious assumptions: [%0.2f, %0.2f]',
        traj[, tail(lb.dual, 1)],
        traj[, tail(ub.dual, 1)]
        )

traj <- read.traj('results/iv_late_just.csv')
sprintf('late bounds with justified assumptions: [%0.2f, %0.2f]',
        traj[, tail(lb.dual, 1)],
        traj[, tail(ub.dual, 1)]
        )



###############################
## noncompliance uncertainty ##
###############################

ci <- fread('results/ci_data.csv')

pop.lb <- ci[1, real_lb]
pop.ub <- ci[1, real_ub]

ci[, V1 := NULL]
ci[, real_lb := NULL]
ci[, real_ub := NULL]
ci[, N := as.numeric(gsub('n', '', basename(dirname(file))))]
ci[, N.label := sprintf('N = %d,000', N / 1000), by = N]
ci[, sim := as.numeric(gsub('\\.csv', '', basename(file)))]
ci[, file := NULL]

ci <- melt(ci,
           id.vars = c('N', 'N.label', 'sim')
           )
ci[, direction := gsub('_(kl|gauss|def)', '', variable)]
ci[, method := gsub('[ul]b_', '', variable)]
ci[,
   method.label := factor(method,
                          levels = c('def',
                                     'kl',
                                     'gauss'
                                     ),
                          labels = c('Estimated\nBounds',
                                     'Bernoulli-KL\nConf. Bounds',
                                     'Asymptotic\nConf. Bounds'
                                     )
                          )
   ]

ci <- dcast(ci,
            sim + N.label + method.label ~ direction,
            value.var = 'value'
            )
ci[, contain.label := factor(lb <= pop.lb & pop.ub <= ub,
                             levels = c(FALSE,
                                        TRUE
                                        ),
                             labels = c('Fails to contain pop. bounds',
                                        'Contains pop. bounds'
                                        )
                             )
   ]

print(
  xtable(
    cbind(
      c('Quantity', 'Lower bound', 'Upper bound'),
      t(
        ci[method.label == 'Estimated\nBounds',
           .(`Lower bound` = sprintf('%0.4f', mean(lb)),
             `Upper bound` = sprintf('%0.4f', mean(ub))
             ),
           by = N.label
           ]
      ),
      c('Population', sprintf('%0.4f', pop.lb), sprintf('%0.4f', pop.ub))
    )
  ),
  include.colnames = FALSE,
  include.rownames = FALSE
)

pdf('results/coverage.pdf', 12, 9)
ggplot(ci) +
  geom_errorbarh(aes(xmin = lb,
                     xmax = ub,
                     y = sim,
                     color = contain.label
                     ),
                 data = ci[method.label != 'Sanov',],
                 height = 0
                 ) +
  geom_errorbarh(aes(xmin = lb,
                     xmax = ub,
                     y = sim,
                     color = contain.label
                     ),
                 data = ci[method.label != 'Sanov' & !ci$contain,],
                 height = 0
                 ) +
  geom_vline(xintercept = pop.lb,
             color = 'gray60',
             linetype = '12',
             size = 2
             ) +
  geom_vline(xintercept = pop.ub,
             color = 'gray60',
             linetype = '12',
             size = 2
             ) +
  geom_vline(xintercept = 0,
             color = 'black',
             linetype = 'dashed',
             size = 1
             ) +
  facet_grid(method.label ~ N.label) +
  scale_color_manual(values = c('Contains pop. bounds' = blue,
                                'Fails to contain pop. bounds' = red
                                ),
                     drop = FALSE,
                     name = NULL
                     ) +
  guides(color = guide_legend(override.aes = list(size = 2))) +
  theme_light(base_size = 24) +
  theme(legend.position = c(.5, -.3),
        legend.direction = 'vertical',
        plot.margin = unit(c(1, 1, 3, 0.5), 'cm')
        ) +
  scale_x_continuous(breaks = c(-1, 0, 1),
                     labels = c('-1', '0', '1'),
                     limits = c(-1, 1)
                     ) +
  scale_y_continuous(breaks = c(1, 1000)) +
  xlab('\nAverage treatment effect') +
  ylab('Simulation')
dev.off()



###############
## selection ##
###############

set.seed(02139)

## ## version 1: check with randomly generated dgp ###

## ## confounded X -> Y: define joint distribution over X & Y response variables
## R.X <- data.table(X = 0:1)
## R.Y <- CJ(Y0 = 0:1,
##           Y1 = 0:1
##           )
## ind.XY <- CJ(ind.X = 1:nrow(R.X),
##              ind.Y = 1:nrow(R.Y)
##              )
## R.XY <- cbind(R.X[ind.XY$ind.X,],
##               R.Y[ind.XY$ind.Y,]
##               )
## ## look up factual outcome from potential outcomes
## R.XY[, Y := ifelse(X == 0, yes = Y0, no = Y1)]
## ## randomly generate a distribution
## R.XY[, p.XY := as.numeric(rdirichlet(1, alpha = rep(1, .N)))]

## ## unconfounded Y -> S: define distribution over S response variables
## R.S <- CJ(S0 = 0:1,
##           S1 = 0:1
##           )
## R.S[, p.S := as.numeric(rdirichlet(1, alpha = rep(1, .N)))]

## ## combine into joint distribution over X, Y, S response variables
## ind.XYS <- CJ(ind.XY = 1:nrow(R.XY),
##               ind.S = 1:nrow(R.S)
##               )
## R.XYS <- cbind(R.XY[ind.XYS$ind.XY,],
##                R.S[ind.XYS$ind.S]
##                )
## ## look up factual outcome from potential outcomes
## R.XYS[, S := ifelse(Y == 0, yes = S0, no = S1)]
## ## generate distribution by combining independent margins
## R.XYS[, p := p.XY * p.S]
## R.XYS[, p.XY := NULL]
## R.XYS[, p.S := NULL]

## ## estimand
## ate <- R.XYS[, sum(p * (Y1 - Y0))]
## naive <-
##   R.XYS[X == 1, sum(p * (Y1 - Y0)) / sum(p)] -
##   R.XYS[X == 0, sum(p * (Y1 - Y0)) / sum(p)]

## ## observed quantities
## obs <- R.XYS[S == 1, .(p = sum(p)), by = c('X', 'Y')]
## obs <- obs[, p := p / sum(p)]
## r <- R.XYS[S == 1, sum(p)]

## ## sample
## N <- 1e6
## sample <- R.XYS[sample.int(.N, size = N, replace = TRUE, prob = p), -'p']
## sate <- sample[, mean(Y1 - Y0)]



### version 2: check with autobounds ###

obs <- fread('data/selection_obsqty.csv')

obs[, p := prob]
obs[, prob := NULL]

## Pr(S == 1)
r <- obs[, sum(p)]

## Pr(X == x, Y == y | S == 1)
obs[, p := p / sum(p)]

## observed quantities used in computing bounds

A.y0s1 <- obs[X == 1 & Y == 0, p] / obs[X == 0 & Y == 0, p]
A.y1s1 <- obs[X == 1 & Y == 1, p] / obs[X == 0 & Y == 1, p]
p.x0y1.s1 <- obs[X == 0 & Y == 1, p]
p.x1y0.s1 <- obs[X == 1 & Y == 0, p]

## ## validation against dgp in eq 16 with sec 7.2 params, tab 3c results
## expit <- function(x) exp(x) / (1 + exp(x))
## N <- 1e6
## U <- rbinom(N, 1, .5)
## Z <- rbinom(N, 1, .5)
## a1 <- b1 <- c1 <- -1
## a2 <- a3 <- b2 <- b3 <- c2 <- .5
## a4 <- b4 <- c3 <- c4 <- 0
## X <- rbinom(N, 1, expit(a1 + a2 * U + a3 * Z + a4 * U * Z))
## Y <- rbinom(N, 1, expit(b1 + b2 * U + b3 * X + b4 * U * X))
## S <- rbinom(N, 1, expit(c1 + c2 * Y + c3 * U + c4 * X))
## r <- mean(S)
## obs <- data.table(X, Y, S)
## obs <- obs[S == 1, .(p = .N), by = c('X', 'Y')]
## obs[, p := p / sum(p)]
## A.y0s1 <- obs[X == 1 & Y == 0, p] / obs[X == 0 & Y == 0, p]
## A.y1s1 <- obs[X == 1 & Y == 1, p] / obs[X == 0 & Y == 1, p]
## p.x0y1.s1 <- obs[X == 0 & Y == 1, p]
## p.x1y0.s1 <- obs[X == 1 & Y == 0, p]
## mean(rbinom(N, 1, expit(b1 + b2 * U + b3 * 1 + b4 * U * 1))) -
##   mean(rbinom(N, 1, expit(b1 + b2 * U + b3 * 0 + b4 * U * 0)))
## ## check when using dgp from sjolander et al
## c(lb = lb, tab.3c = -.50)
## c(ub = ub, tab.3c = .64)

lb.theory <-
  -(p.x0y1.s1 + p.x1y0.s1) * r -
  max(1 / (1 + A.y1s1), A.y0s1 / (1 + A.y0s1)) * (1 - r)
ub.theory <-
  1 - (p.x0y1.s1 + p.x1y0.s1) * r -
  min(1 / (1 + A.y1s1), A.y0s1 / (1 + A.y0s1)) * (1 - r)

traj <- read.traj('results/selection.csv', extend = .025)
lb.autobounds <- traj[, tail(lb.dual, 1)]
round(lb.autobounds, 2)
ub.autobounds <- traj[, tail(ub.dual, 1)]
round(ub.autobounds, 2)

## same result as causaloptim, to tolerance of 1e-4
all.equal(lb.theory, lb.autobounds, tolerance = 1e-4)
all.equal(ub.theory, ub.autobounds, tolerance = 1e-4)

pdf('results/selection_trajectory.pdf',
    width = 12,
    height = 9
    )
plot.bounds(traj, lb.theory, ub.theory)
dev.off()



################################
## model b: measurement error ##
################################

set.seed(02139)

## ## version 1: check with randomly generated dgp ###

## ## unconfounded X -> Y: define joint distribution over X & Y response variables
## R.X <- data.table(X = 0:1)
## R.X[, p.X := as.numeric(rdirichlet(1, alpha = rep(1, .N)))]

## ## confounded Ya -> Yb: define joint distribution over Ya & Yb response variables
## R.YaYb <- CJ(Ya0 = 0:1,
##              Ya1 = 0:1,
##              Yb0 = 0:1,
##              Yb1 = 0:1
##              )
## R.YaYb[, p.YaYb := as.numeric(rdirichlet(1, alpha = rep(1, .N)))]
## ## impose monotonicity: Yb(1) >= Yb(0)
## R.YaYb[Yb1 < Yb0, p.YaYb := 0]
## R.YaYb[, p.YaYb := p.YaYb / sum(p.YaYb)]

## ## combine indep margins into joint distr over X, Ya, Yb response variables
## ind.XYaYb <- CJ(ind.X = 1:nrow(R.X),
##                 ind.YaYb = 1:nrow(R.YaYb)
##                 )
## R.XYaYb <- cbind(R.X[ind.XYaYb$ind.X,],
##                  R.YaYb[ind.XYaYb$ind.YaYb]
##                  )
## ## generate distribution by combining independent margins
## R.XYaYb[, p := p.X * p.YaYb]
## R.XYaYb[, p.X := NULL]
## R.XYaYb[, p.YaYb := NULL]

## ## look up factual outcome from potential outcomes
## R.XYaYb[, Ya := ifelse(X == 0, yes = Ya0, no = Ya1)]
## R.XYaYb[, Yb := ifelse(Ya == 0, yes = Yb0, no = Yb1)]

## ## observed quantities
## obs <- R.XYaYb[, .(p = sum(p)), by = c('X', 'Yb')]



### version 2: check with autobounds ###

obs <- fread('data/measurement_error.csv')  # S is Ystar
obs[, p := prob]
obs[, prob := NULL]

## measurement error bounds from causaloptim examples:
## https://cran.r-project.org/web/packages/causaloptim/vignettes/example-code.html
G <- graph_from_literal(Ul -+ X -+ Y -+ Y2, Ur -+ Y, Ur -+ Y2)
V(G)$leftside <- c(1, 1, 0, 0, 0)
V(G)$latent <- c(1, 0, 1, 0, 1)
V(G)$nvals <- c(Inf, 2, 2, 2, Inf)
E(G)$rlconnect <- c(0, 0, 0, 0, 0)
E(G)$edge.monotone <- c(0, 0, 0, 0, 0)

problem <- analyze_graph(G,
                         constraints = 'Y2(Y = 1) >= Y2(Y = 0)',
                         effectt = 'p{Y(X = 1) = 1} - p{Y(X = 0) = 1}'
                         )
## pa_b: P(Y2 = a | X = b)
bounds <- optimize_effect(problem)
## MAX {
## - 1
## 2 p0_0 - 2 p0_1 - 1
## }
## MIN {
## 2 p0_0 - 2 p0_1 + 1
## 1
## }

## ## verify point identification in an extreme case
## obs[, p := c(.5, 0, 0, .5)]

lb.causaloptim <- max(
  -1,
  2 * obs[X == 0, p[S == 0] / sum(p)] +     # +2 p0_0 = 2 * p(Ystar=0 | X=0)
    -2 * obs[X == 1, p[S == 0] / sum(p)] +  # -2 p0_1 = 2 * p(Ystar=0 | X=1)
      -1
)
ub.causaloptim <- min(
  2 * obs[X == 0, p[S == 0] / sum(p)] +     # +2 p0_0 = 2 * p(Y2=0 | X=0)
    -2 * obs[X == 1, p[S == 0] / sum(p)] +  # -2 p0_1 = 2 * p(Y2=0 | X=1)
      1,
  1
)

traj <- read.traj('results/measurement_error.csv')

## flag duplicated results after convergence
converge.ind <- max(
  min(which(traj$lb.dual == max(traj$lb.dual))),
  min(which(traj$ub.dual == min(traj$ub.dual)))
)
## show a few of these duplicated values to allow visual verification
## that primal & dual values are equal, per reviewer request
converge.ind <- converge.ind + round(converge.ind * .025)
traj <- traj[1:converge.ind]

lb.autobounds <- traj[, tail(lb.dual, 1)]
round(lb.autobounds, 2)
ub.autobounds <- traj[, tail(ub.dual, 1)]
round(ub.autobounds, 2)

## same result as causaloptim, to tolerance of 1e-6

lb.causaloptim
ub.causaloptim

lb.autobounds
ub.autobounds

pdf('results/measurement_error_trajectory.pdf',
    width = 12,
    height = 9
    )
plot.bounds(traj, lb.causaloptim, ub.causaloptim)
dev.off()



#################
## missingness ##
#################

set.seed(02139)

## ## version 1: check with randomly generated dgp ###

## ## unconfounded X: distribution over X response variables
## R.X <- data.table(X = 0:1)
## R.X[, p.X := as.numeric(rdirichlet(1, alpha = rep(1, .N)))]

## ## unconfounded true Y (Ya): distribution over Ya response variables
## R.Ya <- CJ(Ya0 = 0:1,
##            Ya1 = 0:1
##            )
## R.Ya[, p.Ya := as.numeric(rdirichlet(1, alpha = rep(1, .N)))]

## ## unconfounded reporting: distribution over R response variables
## R.R <- CJ(R0 = 0:1,
##           R1 = 0:1
##           )
## R.R[, p.R := as.numeric(rdirichlet(1, alpha = rep(1, .N)))]

## ind.XYaR <- CJ(ind.X = 1:nrow(R.X),
##                ind.Ya = 1:nrow(R.Ya),
##                ind.R = 1:nrow(R.R)
##                )
## R.XYaRYb <- cbind(R.X[ind.XYaR$ind.X,],
##                   R.Ya[ind.XYaR$ind.Ya,],
##                   R.R[ind.XYaR$ind.R]
##                   )
## ## look up factual outcome from potential outcomes
## R.XYaRYb[, Ya := ifelse(X == 0, yes = Ya0, no = Ya1)]
## R.XYaRYb[, R := ifelse(Ya == 0, yes = R1, no = R0)]
## R.XYaRYb[, Yb := ifelse(R == 0, yes = NA, no = Ya1)]
## ## generate distribution by combining independent margins
## R.XYaRYb[, p := p.X * p.Ya * p.R]
## R.XYaRYb[, p.X := NULL]
## R.XYaRYb[, p.Ya := NULL]
## R.XYaRYb[, p.R := NULL]

## ## estimand
## ate <- R.XYaRYb[, sum(p * (Ya1 - Ya0))]
## naive <-
##   R.XYaRYb[X == 1 & !is.na(Yb), sum(p * Yb) / sum(p)] -
##   R.XYaRYb[X == 0 & !is.na(Yb), sum(p * Yb) / sum(p)]

## obs <- R.XYaRYb[, .(p = sum(p)), by = c('X', 'Yb')]

## ## numerical validation against alternative implementation
## N <- 1e6
## sample <- R.XYaRYb[sample.int(.N, size = N, replace = TRUE, prob = p), -'p']
## sate <- sample[, mean(Ya1 - Ya0)]
## mean(sample[X == 1, Yb], na.rm = TRUE) - mean(sample[X == 0, Yb], na.rm = TRUE)
## c(lb = lb,
##   sim.lb =
##     sample[X == 1, mean(ifelse(is.na(Yb), 0, Yb))] -
##     sample[X == 0, mean(ifelse(is.na(Yb), 1, Yb))]
##   )
## c(ub = ub,
##   sim.ub =
##     sample[X == 1, mean(ifelse(is.na(Yb), 1, Yb))] -
##     sample[X == 0, mean(ifelse(is.na(Yb), 0, Yb))]
##   )



### version 2: check with autobounds ###

obs.xr <- fread('data/manski1_2.csv')
p.r1 <- obs.xr[R == 1, sum(prob)]

obs.xry.cond.r1 <- fread('data/manski1_1.csv')
obs.xry <- copy(obs.xry.cond.r1)
obs.xry[, prob := prob * p.r1]

obs <- rbind(obs.xr[R == 0,],
             obs.xry,
             fill = TRUE
             )
obs[, p := prob]
obs[, prob := NULL]

lb.theory <-
  obs[X == 1, sum(p * ifelse(is.na(Y), yes = 0, no = Y)) / sum(p)] -
  obs[X == 0, sum(p * ifelse(is.na(Y), yes = 1, no = Y)) / sum(p)]

ub.theory <-
  obs[X == 1, sum(p * ifelse(is.na(Y), yes = 1, no = Y)) / sum(p)] -
  obs[X == 0, sum(p * ifelse(is.na(Y), yes = 0, no = Y)) / sum(p)]



traj.couenne <- read.traj('results/nonresponse.csv')
lb.autobounds.couenne <- traj.couenne[, tail(lb.dual, 1)]
ub.autobounds.couenne <- traj.couenne[, tail(ub.dual, 1)]

clean.scip.log('results/nonresponse_min.log')
clean.scip.log('results/nonresponse_max.log')

traj.scip <- read.traj.scip('results/nonresponse_min_clean.log',
                            'results/nonresponse_max_clean.log'
                            )
lb.autobounds.scip <- traj.scip[, tail(lb.dual, 1)]
ub.autobounds.scip <- traj.scip[, tail(ub.dual, 1)]



## same result as theory, to tolerance of 1e-2

lb.theory
ub.theory

lb.autobounds.couenne
ub.autobounds.couenne

lb.autobounds.scip
ub.autobounds.scip

pdf('./results/manski1.pdf',
    width = 12,
    height = 9
    )
plot.bounds(traj.scip, lb.theory, ub.theory)
dev.off()



#######################
## joint missingness ##
#######################

set.seed(02139)

obs <- fread('data/manski2_obs.csv')
obs[, p := prob]
obs[, prob := NULL]

manski.lb.xyNAtox0 <- copy(obs)
manski.lb.xyNAtox1 <- copy(obs)
manski.ub.xyNAtox0 <- copy(obs)
manski.ub.xyNAtox1 <- copy(obs)

## lb: set ctrl (treated) units with y=NA to y=1 (y=0)
manski.lb.xyNAtox0[Xobs== 0 & Yobs == 'M', Yobs := 1]
manski.lb.xyNAtox0[Xobs== 1 & Yobs == 'M', Yobs := 0]
manski.lb.xyNAtox1[Xobs== 0 & Yobs == 'M', Yobs := 1]
manski.lb.xyNAtox1[Xobs== 1 & Yobs == 'M', Yobs := 0]
## ub: set ctrl (treated) units with y=NA to y=0 (y=1)
manski.ub.xyNAtox0[Xobs== 0 & Yobs == 'M', Yobs := 0]
manski.ub.xyNAtox0[Xobs== 1 & Yobs == 'M', Yobs := 1]
manski.ub.xyNAtox1[Xobs== 0 & Yobs == 'M', Yobs := 0]
manski.ub.xyNAtox1[Xobs== 1 & Yobs == 'M', Yobs := 1]

## lb: set y=0 (y=1) units with x=NA to treated (ctrl)
manski.lb.xyNAtox0[Yobs == 0 & Xobs == 'M', Xobs:= 1]
manski.lb.xyNAtox0[Yobs == 1 & Xobs == 'M', Xobs:= 0]
manski.lb.xyNAtox1[Yobs == 0 & Xobs == 'M', Xobs:= 1]
manski.lb.xyNAtox1[Yobs == 1 & Xobs == 'M', Xobs:= 0]
## ub: set y=0 (y=1) units with x=NA to ctrl (treated)
manski.lb.xyNAtox0[Yobs == 0 & Xobs == 'M', Xobs:= 0]
manski.lb.xyNAtox0[Yobs == 1 & Xobs == 'M', Xobs:= 1]
manski.lb.xyNAtox1[Yobs == 0 & Xobs == 'M', Xobs:= 0]
manski.lb.xyNAtox1[Yobs == 1 & Xobs == 'M', Xobs:= 1]

## lb version 1: take x=y=NA units and set to x=0, y=1
manski.lb.xyNAtox0[Xobs == 'M' & Yobs == 'M', `:=`(Xobs= 0, Yobs = 1)]
## lb version 2: take x=y=NA units and set to x=1, y=0
manski.lb.xyNAtox1[Xobs == 'M' & Yobs == 'M', `:=`(Xobs= 1, Yobs = 0)]

## ub version 1: take x=y=NA units and set to x=0, y=0
manski.ub.xyNAtox0[Xobs == 'M' & Yobs == 'M', `:=`(Xobs= 0, Yobs = 0)]
## ub version 2: take x=y=NA units and set to x=1, y=1
manski.ub.xyNAtox1[Xobs == 'M' & Yobs == 'M', `:=`(Xobs= 1, Yobs = 1)]



### E[Y|X=1] - E[Y|X=0] ###

manski.lb.v1 <-
  (manski.lb.xyNAtox0[Xobs== 1 & Yobs == 1, sum(p)] /
     manski.lb.xyNAtox0[Xobs== 1, sum(p)]
  ) -
  (manski.lb.xyNAtox0[Xobs== 0 & Yobs == 1, sum(p)] /
     manski.lb.xyNAtox0[Xobs== 0, sum(p)]
  )
manski.lb.v2 <-
  (manski.lb.xyNAtox1[Xobs== 1 & Yobs == 1, sum(p)] /
     manski.lb.xyNAtox1[Xobs== 1, sum(p)]
  ) -
  (manski.lb.xyNAtox1[Xobs== 0 & Yobs == 1, sum(p)] /
     manski.lb.xyNAtox1[Xobs== 0, sum(p)]
  )

manski.ub.v1 <-
  (manski.ub.xyNAtox0[Xobs== 1 & Yobs == 1, sum(p)] /
     manski.ub.xyNAtox0[Xobs== 1, sum(p)]
  ) -
  (manski.ub.xyNAtox0[Xobs== 0 & Yobs == 1, sum(p)] /
     manski.ub.xyNAtox0[Xobs== 0, sum(p)]
  )
manski.ub.v2 <-
  (manski.ub.xyNAtox1[Xobs== 1 & Yobs == 1, sum(p)] /
     manski.ub.xyNAtox1[Xobs== 1, sum(p)]
  ) -
  (manski.ub.xyNAtox1[Xobs== 0 & Yobs == 1, sum(p)] /
     manski.ub.xyNAtox1[Xobs== 0, sum(p)]
  )

lb.manski <- min(manski.lb.v1, manski.lb.v2)
ub.manski <- max(manski.ub.v1, manski.ub.v2)



clean.scip.log('results/nonresponse2_min.log')
clean.scip.log('results/nonresponse2_max.log')

traj.scip <- read.traj.scip('results/nonresponse2_min_clean.log',
                            'results/nonresponse2_max_clean.log'
                            )
lb.autobounds.scip <- traj.scip[, tail(lb.dual, 1)]
ub.autobounds.scip <- traj.scip[, tail(ub.dual, 1)]


lb.theory
ub.theory

lb.autobounds.scip
ub.autobounds.scip



pdf('results/nonresponse2.pdf',
    width = 12,
    height = 9
    )
plot.bounds(traj.scip, lb.manski, ub.manski)
dev.off()
