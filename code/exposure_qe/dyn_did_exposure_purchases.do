// ******************************************
// Author : Gielle Labrador Badia
// Date 12/06/2023

// ******************************************


local winstat = 1

if `winstat' ==0{
	glo path_wrk = ""
}
else {
	glo path_wrk = "V:/houde/mortgages/QE_Covid"
}

glo data_path = "$path_wrk/data/data_auction/clean_data"


cd "$data_path"

glo path_graph = "$path_wrk/results/figures"

import delimited "ob_fed_exposure_measure.csv", clear 

local format_plot = "pdf"
// ******************************************
encode firstmonthyear, gen(t)
gen zero = 0

// ******************************************
local controls = ""
local sample = ""
local fe = "hedgeclientkey#t committedinvestorkey coupon#t"


// * Using exposure dummy, exposure_it


local yvar = "loanamount_mean"

reghdfe `yvar'  zero i.exposure##i.t `controls'  `sample', absorb(`fe') vce(robust)
est store est1


coefplot (est1, keep(zero 1.exposure#*) pstyle(p1) recast(connected) ciopts(recast(rcap))), yline(0, lcolor(black) lpattern(dash))  omitted vertical nooffsets title("`addlabel'" ) ytitle(`ylabel') xtitle () graphregion(fcolor(white) lcolor(none)) bgcolor(white) plotregion(fcolor(white) lcolor(none)) xlabel( 1 "Jul 19" 4 "Oct 19" 7 "Jan 20" 10 "April 20" 13 "Jul 20" 16 "Oct 20" 19 "Jan 21" 22 "April 21" ,labsize(small) angle(45))  xline(9, lpattern(dash) lcolor(black))  // xlabel(1 "-1" 2 "0" 3 "1" 4 "2" 5 "3" 6 "4" 7 "5" , labsize(small) )

graph export "$path_graph/did_loan_amount_exposure_dummy.`format_plot'", as(`format_plot') name("Graph") replace

// esttab est1,se r2 ar2 scalar(F)    keep(1.exposure#*)

// ******************************************
local yvar = "winner_bid_max"


reghdfe `yvar' zero i.exposure##i.t `controls'  `sample', absorb(`fe') vce(robust)
est store est2

coefplot (est2, keep(zero 1.exposure#*) pstyle(p1) recast(connected) ciopts(recast(rcap))), yline(0, lcolor(black) lpattern(dash))  omitted vertical nooffsets title("`addlabel'" ) ytitle(`ylabel') xtitle () graphregion(fcolor(white) lcolor(none)) bgcolor(white) plotregion(fcolor(white) lcolor(none)) xlabel( 1 "Jul 19" 4 "Oct 19" 7 "Jan 20" 10 "April 20" 13 "Jul 20" 16 "Oct 20" 19 "Jan 21" 22 "April 21" ,labsize(small) angle(45))  xline(9, lpattern(dash) lcolor(black))  // xlabel(1 "-1" 2 "0" 3 "1" 4 "2" 5 "3" 6 "4" 7 "5" , labsize(small) )

graph export "$path_graph/did_winner_bid_exposure_dummy.`format_plot'", as(`format_plot') name("Graph") replace

// ******************************

// * Using exposure dummy, exposure_i
local yvar = "loanamount_mean"

reghdfe `yvar' zero i.exposure_march_2020##i.t `controls'  `sample', absorb(`fe') vce(robust)
est store est1


coefplot (est1, keep(zero 1.exposure_march_2020#*) pstyle(p1) recast(connected) ciopts(recast(rcap))), yline(0, lcolor(black) lpattern(dash))  omitted vertical nooffsets title("`addlabel'" ) ytitle(`ylabel') xtitle () graphregion(fcolor(white) lcolor(none)) bgcolor(white) plotregion(fcolor(white) lcolor(none)) xlabel( 1 "Jul 19" 4 "Oct 19" 7 "Jan 20" 10 "April 20" 13 "Jul 20" 16 "Oct 20" 19 "Jan 21" 22 "April 21" ,labsize(small) angle(45))  xline(9, lpattern(dash) lcolor(black))  // xlabel(1 "-1" 2 "0" 3 "1" 4 "2" 5 "3" 6 "4" 7 "5" , labsize(small) )

graph export "$path_graph/did_loan_amount_exposure_march_dummy.`format_plot'", as(`format_plot') name("Graph") replace


// esttab est1,se r2 ar2 scalar(F)    keep(1.exposure#*)

// ******************************************
local yvar = "winner_bid_max"


reghdfe `yvar' zero  i.exposure_march_2020##i.t `controls'  `sample', absorb(`fe') vce(robust)
est store est2

coefplot (est2, keep(zero 1.exposure_march_2020#*) pstyle(p1) recast(connected) ciopts(recast(rcap))), yline(0, lcolor(black) lpattern(dash))  omitted vertical nooffsets title("`addlabel'" ) ytitle(`ylabel') xtitle () graphregion(fcolor(white) lcolor(none)) bgcolor(white) plotregion(fcolor(white) lcolor(none)) xlabel( 1 "Jul 19" 4 "Oct 19" 7 "Jan 20" 10 "April 20" 13 "Jul 20" 16 "Oct 20" 19 "Jan 21" 22 "April 21" ,labsize(small) angle(45))  xline(9, lpattern(dash) lcolor(black))  // xlabel(1 "-1" 2 "0" 3 "1" 4 "2" 5 "3" 6 "4" 7 "5" , labsize(small) )

graph export "$path_graph/did_winner_bid_exposure_march_dummy.`format_plot'", as(`format_plot') name("Graph") replace



// esttab est1,se r2 ar2 scalar(F)  keep(1.exposure#*)

//
//
// // ******************************
//
// * Using exposure dummy, exposure_amount_it

local yvar = "loanamount_mean"

reghdfe `yvar' zero  c.fed_trade_amount##i.t `controls'  `sample', absorb(`fe') vce(robust)
est store est1

// esttab est1,se r2 ar2 scalar(F)    keep(1.exposure#*)

coefplot (est1, keep(zero *#*) pstyle(p1) recast(connected) ciopts(recast(rcap))), yline(0, lcolor(black) lpattern(dash))  omitted vertical nooffsets title("`addlabel'" ) ytitle(`ylabel') xtitle () graphregion(fcolor(white) lcolor(none)) bgcolor(white) plotregion(fcolor(white) lcolor(none)) xlabel( 1 "Jul 19" 4 "Oct 19" 7 "Jan 20" 10 "April 20" 13 "Jul 20" 16 "Oct 20" 19 "Jan 21" 22 "April 21" ,labsize(small) angle(45))  xline(9, lpattern(dash) lcolor(black))  // xlabel(1 "-1" 2 "0" 3 "1" 4 "2" 5 "3" 6 "4" 7 "5" , labsize(small) )

graph export "$path_graph/did_loan_amount_expamount.`format_plot'", as(`format_plot') name("Graph") replace

// ******************************************
local yvar = "winner_bid_max"


reghdfe `yvar' zero  c.fed_trade_amount##i.t `controls'  `sample', absorb(`fe') vce(robust)
est store est2

coefplot (est2, keep(zero *#*) pstyle(p1) recast(connected) ciopts(recast(rcap))), yline(0, lcolor(black) lpattern(dash))  omitted vertical nooffsets title("`addlabel'" ) ytitle(`ylabel') xtitle () graphregion(fcolor(white) lcolor(none)) bgcolor(white) plotregion(fcolor(white) lcolor(none)) xlabel( 1 "Jul 19" 4 "Oct 19" 7 "Jan 20" 10 "April 20" 13 "Jul 20" 16 "Oct 20" 19 "Jan 21" 22 "April 21" ,labsize(small) angle(45))  xline(9, lpattern(dash) lcolor(black))  // xlabel(1 "-1" 2 "0" 3 "1" 4 "2" 5 "3" 6 "4" 7 "5" , labsize(small) )

graph export "$path_graph/did_winner_bid_expamount.`format_plot'", as(`format_plot') name("Graph") replace


// ******************************

// * Using exposure dummy, exposure_amount_i

local yvar = "loanamount_mean"

reghdfe `yvar' zero c.fed_trade_amount_march_2020##i.t `controls'  `sample', absorb(`fe') vce(robust)
est store est1

coefplot (est1, keep(zero *#*) pstyle(p1) recast(connected) ciopts(recast(rcap))), yline(0, lcolor(black) lpattern(dash))  omitted vertical nooffsets title("`addlabel'" ) ytitle(`ylabel') xtitle () graphregion(fcolor(white) lcolor(none)) bgcolor(white) plotregion(fcolor(white) lcolor(none)) xlabel( 1 "Jul 19" 4 "Oct 19" 7 "Jan 20" 10 "April 20" 13 "Jul 20" 16 "Oct 20" 19 "Jan 21" 22 "April 21" ,labsize(small) angle(45))  xline(9, lpattern(dash) lcolor(black))  // xlabel(1 "-1" 2 "0" 3 "1" 4 "2" 5 "3" 6 "4" 7 "5" , labsize(small) )

graph export "$path_graph/did_loan_amount_expamount_march_dummy.`format_plot'", as(`format_plot') name("Graph") replace


// esttab est1,se r2 ar2 scalar(F)    keep(1.exposure#*)

// ******************************************
local yvar = "winner_bid_max"


reghdfe `yvar' zero  c.fed_trade_amount_march_2020##i.t `controls'  `sample', absorb(`fe') vce(robust)
est store est2

coefplot (est2, keep(zero *#*) pstyle(p1) recast(connected) ciopts(recast(rcap))), yline(0, lcolor(black) lpattern(dash))  omitted vertical nooffsets title("`addlabel'" ) ytitle(`ylabel') xtitle () graphregion(fcolor(white) lcolor(none)) bgcolor(white) plotregion(fcolor(white) lcolor(none)) xlabel( 1 "Jul 19" 4 "Oct 19" 7 "Jan 20" 10 "April 20" 13 "Jul 20" 16 "Oct 20" 19 "Jan 21" 22 "April 21" ,labsize(small) angle(45))  xline(9, lpattern(dash) lcolor(black))  // xlabel(1 "-1" 2 "0" 3 "1" 4 "2" 5 "3" 6 "4" 7 "5" , labsize(small) )

graph export "$path_graph/did_winner_bid_expamount_march_dummy.`format_plot'", as(`format_plot') name("Graph") replace



// 