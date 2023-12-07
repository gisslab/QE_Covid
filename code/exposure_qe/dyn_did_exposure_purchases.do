// ******************************************
// Author : Gielle Labrador Badia
// Date 12/06/2023

// ******************************************


local winstat = 1

if `winstat' ==0{
	glo path_wrk = ""
}
else {
	glo path_wrk = "V:/project/houde/mortgages/QE_Covid"
}

glo data_path = "$path_wrk/data/data_auction/clean_data"


cd "$data_path"

import ilimited "ob_fed_exposure_measure.csv", clear 
// ******************************************

local yvar = "num_sale"

local controls = ""
local sample = ""
local fe = ""


reghdfe yvar  i.exposure##ib0.firstmonthyear `controls'  `sample', absorb(`fe') vce(robust)
est store num_sale

