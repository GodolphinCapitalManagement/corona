
import warnings
warnings.filterwarnings("ignore")
import sys
import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
sns.set()

plt.rcParams.update({
    "font.family": "Source Sans Pro",
    "font.serif": ["Source Sans Pro"], 
    "font.sans-serif": ["Source Sans Pro"],
    "font.size": 10,
})

import pathlib
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import pytoml

from fredapi import Fred

import feather

import pymc3 as pm
import arviz as az

import lifelines
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from scipy.special import expit

from analytics import utils

models_dir = "/home/gsinha/admin/db/dev/Python/projects/models/"
import_dir = models_dir + "defers/"
sys.path.append(import_dir)
from common import *

data_dir = models_dir + "data/"

omap = {"LC": "I", "PR": "II"}

results_dir = {
  "LC": models_dir + "defers/pymc3/" + "originator_" + omap["LC"] + "/",
  "PR": models_dir + "defers/pymc3/" + "originator_" + omap["PR"] + "/"
}

idx = pd.IndexSlice

ASOF_DATE = datetime.date(2020, 6, 8)

fname = data_dir + "claims.pkl"
with open(fname, "rb") as f:
    claims_dict = joblib.load(f)


def read_results(model_type, originator, asof_date):
    ''' read pickled results '''

    import_dir = models_dir + "defers/"
    fname = (
        import_dir + "pymc3/originator_" + originator + "/results/" + 
        "_".join(["defer", originator, model_type, asof_date.isoformat()])
      )
    fname += ".pkl"

    with open(fname, "rb") as f:
        out_dict = joblib.load(f)

    return out_dict

out_dict = {}
pipe_dict = {}
scaler_dict = {}
obs_covars_dict = {}
hard_df_dict = {}
test_dict = {}

for i in [omap["LC"], omap["PR"]]:
    for j in ["pooled", "hier"]:
        out_dict[":".join([i, j])] = read_results(j, i, ASOF_DATE)
        
    orig_model_key = ":".join([i, "hier"])
    pipe_dict[i] = {
        "stage_one": out_dict[orig_model_key]["pipe"]["p_s_1"],
        "stage_two": out_dict[orig_model_key]["pipe"]["p_s_2"],
        "stage_three": out_dict[orig_model_key]["pipe"]["p_s_3"],
        "stage_four": out_dict[orig_model_key]["pipe"]["p_s_4"],
    }
    scaler_dict[i] = (
        pipe_dict[i]["stage_two"].named_steps.std_dummy.numeric_transformer.named_steps["scaler"]
    )
    obs_covars_dict[i] = out_dict[orig_model_key]["obs_covars"]
    hard_df_dict[i] = out_dict[orig_model_key]["hard_df"]
    test_dict[i] = out_dict[orig_model_key]["test"]

risk_df = gen_labor_risk_df(
    "articles_spreadsheet_extended.xlsx", data_dir
)


def make_az_data(originator, model_type):
    ''' make az data instance for originator '''

    orig_model_key = ":".join([originator, model_type])
    model = out_dict[orig_model_key]["model"]
    trace = out_dict[orig_model_key]["trace"]

    pipe_stage_two = pipe_dict[originator]["stage_two"]
    pipe_stage_three = pipe_dict[originator]["stage_three"]
    pipe_stage_four = pipe_dict[originator]["stage_four"]
  
    t_covars = pipe_stage_four.named_steps.spline.colnames

    if model_type == "pooled":
        b_names = ["γ"] + t_covars + pipe_stage_two.named_steps.std_dummy.col_names
        az_data = az.from_pymc3(trace=trace, model=model, coords={'covars': b_names}, dims={'b': ['covars']})
        st_out = pd.DataFrame()

        b_out = az.summary(az_data, round_to=3, var_names=["b"])
        b_out.index = b_names
    else:
        state_fips_indexes_df = pipe_stage_three.named_steps.hier_index.grp_0_grp_1_indexes_df
        index_0_to_st_code_df = state_fips_indexes_df.drop_duplicates(subset=["st_code"])[
        ["index_0", "st_code"]].set_index("index_0")

        index_0_to_st_code_df = pd.merge(index_0_to_st_code_df, states_df, on="st_code")
        b_names = pipe_stage_two.named_steps.std_dummy.col_names[:-1]
        c_names = ["γ"] + t_covars +  ["η"]
        az_data = az.from_pymc3(
            trace=trace, model=model, 
            coords={'obs_covars': b_names, "pop_covars": c_names, 'st_code': index_0_to_st_code_df.state.to_list()},
            dims={'b': ['obs_covars'], "g_c_μ": ["pop_covars"], "g_c_σ": ["pop_covars"], "st_c_μ": ["st_code"], "st_c_μ_σ": ["st_code"]}
        )
        st_out = az.summary(az_data, var_names=["st_c_μ"], round_to=3)
        st_out_idx = pd.MultiIndex.from_tuples(
            [(x, y) for x in index_0_to_st_code_df.state.to_list() for y in c_names],
            names=["state", "param"]
        )
        st_out.index = st_out_idx

        b_out = az.summary(az_data, round_to=3, var_names=["b"])
        b_out.index = b_names

    return trace, az_data, st_out, b_out

# just need one issuer for the combined data
hard_df = hard_df_dict[omap["LC"]]

ic_date = (
  pipe_dict[omap["LC"]]["stage_one"].named_steps.add_state_macro_vars.ic_long_df["edate"].max().date()
)

numeric_features = [
    "fico", "original_balance", "dti", "stated_monthly_income", "age", "pct_ic"
]
categorical_features = [
    "grade", "purpose", "employment_status", "term", "home_ownership", "is_dq"
]

knots = np.linspace(0., 15., 7)

data_scaler_dict = {}
for i in [omap["LC"], omap["PR"]]:
    data_scaler_dict[i] = {
        "mu" : dict(zip(numeric_features, scaler_dict[i].mean_)),
        "sd":  dict(zip(numeric_features, scaler_dict[i].scale_))  
    }


hard_df["current_balance"] = (
  hard_df["original_balance"] * hard_df["cur_note_amount"]/hard_df["note_amount"]
)
hard_df["defer_dollar"] = hard_df["defer"] * hard_df["current_balance"]

def wavg(x):
    return np.average(
        x, weights=hard_df.loc[x.index, "current_balance"]
    )

aaa = hard_df.groupby(["originator", "grade"]).agg(
    n=('loan_id', "count"),
    original_balance=('original_balance', sum),
    current_balance=('current_balance', sum),
    wac=('original_rate', wavg),
    age=('age', wavg),
    fico=('fico', wavg),
    term=('original_term', wavg),
    defer=('defer', wavg),
)

bbb = hard_df.groupby(["originator"]).agg(
    n=('loan_id', "count"),
    original_balance=('original_balance', sum),
    current_balance=('current_balance', sum),
    wac=('original_rate', wavg),
    age=('age', wavg),
    fico=('fico', wavg),
    term=('original_term', wavg),
    defer=('defer', wavg),
)

bbb.index = pd.MultiIndex.from_tuples(
    [(omap["LC"], 'ALL'), (omap["PR"], 'ALL')], names=['originator', 'grade']
)

aaa = pd.concat([aaa, bbb])

ccc = pd.concat(
    [
        pd.Series(hard_df["loan_id"].apply("count"), name="n"),
        pd.Series(hard_df["original_balance"].sum(), name="original_balance"),
        pd.Series(hard_df["current_balance"].sum(), name="current_balance"),
        hard_df[["original_rate", "age", "fico", "original_term"]].apply(wavg).to_frame().T.rename(
            columns={"original_term": "term", "original_rate": "wac"}),
        pd.Series(wavg(hard_df["defer"]), name="defer"),
    ], axis=1
)
ccc.index = [('ALL', 'ALL')]

ddd = pd.concat([aaa, ccc])
ddd["pct"] = ddd["current_balance"]/ddd.loc[pd.IndexSlice["ALL", "ALL"],  "current_balance"]
ddd.index.names = ["Originator", "Grade"]

cfmt = "".join(["r"] * (ddd.shape[1] + 2))
header = [
  "N", "Orig. Bal.", "Cur. Bal.", "WAC", "WALA", "FICO", 
  "WAOT", "Defer", "Share",
]
tbl_fmt = {
  "original_balance": utils.dollar,
  "current_balance": utils.dollar,
  "n": utils.integer, "fico": utils.number,
  "term": utils.number, "age": utils.number,
  "pct": utils.percent, "defer": utils.percent,
  "wac": utils.percent
}

print(
    ddd.to_latex(
      index=True, multirow=True, 
      header=header,
      
      formatters=tbl_fmt,
      column_format=cfmt,
      multicolumn_format="r",
    ))

one_line = ddd.loc[pd.IndexSlice["ALL", :], :]

pos = []
for i in [omap["LC"], omap["PR"]]:
    pos.append(get_due_day(i, ASOF_DATE))
pos_df = pd.concat(pos, ignore_index=True)
pos_df = pos_df[pos_df["loan_id"].isin(hard_df["loan_id"].to_list())]

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharey=True)
for i, v in enumerate([omap["LC"], omap["PR"]]):
    df = pos_df[pos_df["originator"] == v]
    ax[i].hist(df.pmt_day)
    ax[i].set_xlabel("Due day")
    ax[i].set_ylabel("Frequency")
    ax[i].set_title(f"Originator: {v}")
    
plt.tight_layout()

purpose_tbl = summary_by_group(
    ["originator", "purpose"], hard_df
)
purpose_tbl.index.names = ["Originator", "Purpose"]
cfmt = "".join(["r"] * (purpose_tbl.shape[1] + 2))

print(
    purpose_tbl.to_latex(
      index=True, multirow=True, 
      header=header,
      
      formatters=tbl_fmt,
      column_format=cfmt,
      multicolumn_format="r",
    ))

emp_tbl = summary_by_group(
    ["originator", "employment_status"], hard_df
)
emp_tbl = emp_tbl.fillna(0)
emp_tbl.index.names = ["Originator", "Employment"]
cfmt = "".join(["r"] * (emp_tbl.shape[1] + 2))
print(
    emp_tbl.to_latex(
      index=True, multirow=True, 
      header=header,
      
      formatters=tbl_fmt,
      column_format=cfmt,
      multicolumn_format="r",
    ))

homeowner_tbl = summary_by_group(
    ["originator", "home_ownership"], hard_df
)
homeowner_tbl.index.names = ["Originator", "Homeownership"]
cfmt = "".join(["r"] * (homeowner_tbl.shape[1] + 2))
print(
    homeowner_tbl.to_latex(
      index=True, multirow=True, 
      header=header,
      
      formatters=tbl_fmt,
      column_format=cfmt,
      multicolumn_format="r",
    ))

term_tbl = summary_by_group(
    ["originator", "original_term"], hard_df
)
term_tbl.index.names = ["Originator", "Term"]
cfmt = "".join(["r"] * (term_tbl.shape[1] + 2))
print(
    term_tbl.to_latex(
      index=True, multirow=True, 
      header=header,
      
      formatters=tbl_fmt,
      column_format=cfmt,
      multicolumn_format="r",
    ))

dq_tbl = summary_by_group(
    ["originator", "dq_grp"], hard_df
)
dq_tbl.index.names = ["Originator", "DQ Status"]
cfmt = "".join(["r"] * (dq_tbl.shape[1] + 2))
print(
    dq_tbl.to_latex(
      index=True, multirow=True, 
      header=header,
      
      formatters=tbl_fmt,
      column_format=cfmt,
      multicolumn_format="r",
    ))


stage_one_df = pipe_dict[omap["LC"]]["stage_one"].fit_transform(hard_df)

fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))

sns.boxplot(stage_one_df.sdate.dt.date, stage_one_df.pct_ic, ax=ax[0])
sns.distplot(stage_one_df.pct_ic, ax=ax[1], kde=False)

ax[0].set_xlabel("Week ending: "); 
ax[0].set_ylabel("Weekly claims/Labor Force")
ax[1].set_xlabel("Weekly claims/Labor Force")
ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()

def plot_defer_hazard_by_state(originator):
  ''' plots deferment hazards by state and week '''
  zzz = pipe_dict[originator]["stage_one"].fit_transform(hard_df)
  zzz.reset_index(inplace=True)

  a_df = zzz.groupby(["state"]).agg(
    n=("loan_id", "count"), k=("defer", np.sum), defer=("defer", np.mean),
    pct_ic=("pct_ic", np.mean)
  ).reset_index()

  g = sns.FacetGrid(
    data=a_df.reset_index(),
  )
  g.map(sns.regplot, "pct_ic", "defer", ci=True)
  g.ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
  g.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

  # add annotations one by one with a loop
  for line in range(0, a_df.shape[0]):
    g.ax.text(
      a_df["pct_ic"][line]+0.001, a_df["defer"][line], a_df["state"][line], 
      horizontalalignment='left', size='medium', color='red', 
      weight='semibold', alpha=0.25
    )

  g.ax.figure.set_size_inches(15, 8)
  g.ax.set_xlabel("Weekly claims/March Employment")
  g.ax.set_ylabel("Deferment hazard")

  return g

g = plot_defer_hazard_by_state(omap["LC"])
sns.despine(left=True)

g = plot_defer_hazard_by_state(omap["PR"])
sns.despine(left=True)

def create_state_aggs(originator, hard_df, risk_df):
    ''' pct deferment vs pct low risk '''
    
    xbar_df = hard_df.groupby(["originator", "state"]).agg(
      n=("loan_id", "count"), k=('defer', np.sum),
      pct=('defer', np.mean), balance=('cur_note_amount', sum)
    ).loc[pd.IndexSlice[originator, :], :].droplevel(0).reset_index()
    
    xbar_df = pd.merge(xbar_df, risk_df, on="state")
    
    return xbar_df


T = hard_df.dur
E = hard_df.defer

bandwidth = 1
naf = NelsonAalenFitter()
lc = hard_df["originator"].isin([omap["LC"]])

naf.fit(T[lc],event_observed=E[lc], label="Originator I")
ax = naf.plot_hazard(bandwidth=bandwidth, figsize=(10, 5))

naf.fit(T[~lc], event_observed=E[~lc], label="Originator II")
naf.plot_hazard(ax=ax, bandwidth=bandwidth)

ax.set_xlabel("Weeks since March 14th, 2020")
ax.set_ylabel("Weekly hazard")

_  = plt.xlim(0, hard_df.dur.max() + 1)

lt_df = lifelines.utils.survival_table_from_events(
    hard_df.dur, 
    hard_df.defer, collapse=True
)
print(lt_df.to_latex(column_format='rrrrr'))


def make_ppc_plot(originator):
    ''' make ppc plot '''
    
    orig_model_key = ":".join([originator, "hier"])
    ppc, s_1_df, _ = simulate(
        test_dict[originator], claims_dict["chg_df"], originator,
        ASOF_DATE, out_dict[orig_model_key]
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(ppc.mean(axis=0), bins=19, alpha=0.5)
    ax.axvline(s_1_df["defer"].mean())

    ax.set(xlabel='Deferment hazard.', ylabel='Frequency')
  
    pctile = np.percentile(ppc.mean(axis=0), q=[5, 95])
    ax.axvline(pctile[0], color="red", linestyle=":")
    ax.axvline(pctile[1], color="red", linestyle=":")

    _ = ax.text(
      1.65 * s_1_df["defer"].mean(), 0.85 * ax.get_ylim()[1], 
      f'95% HPD: [{pctile[0]:.3f}, {pctile[1]:.3f}]'
    )

    return fig


pooled_trace, pooled_data, _, pooled_b_out = make_az_data(omap["LC"], "pooled")
print(
  pooled_b_out[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat"]].to_latex(
     column_format="rrrrrr"
  )
)

fig = make_ppc_plot(omap["LC"])
fig.show()

hier_trace, hier_data, lc_hier_st_out, lc_hier_b_out = make_az_data(
  omap["LC"], "hier"
)
dff_η = lc_hier_st_out.loc[idx[:, "η"], "mean"].droplevel(level=1).reset_index().rename(
    columns={"mean": "value"}
)

us_states = gpd.read_file(
  "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"
)
merged_us_states_η = pd.merge(
  us_states, dff_η, left_on="STUSPS", right_on="state", how="right")

fig, ax = plt.subplots(1, figsize=(15, 18))
albers_epsg = 2163
ax = us_states[~us_states["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    ax=ax, linewidth=0.25, edgecolor='white', color='grey'
)

ax = merged_us_states_η[~merged_us_states_η["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    column='value', ax=ax, cmap='viridis', scheme="quantiles",  legend=True, 
    legend_kwds={"loc": "upper center", "ncol": 3}
)
_ = ax.axis('off');


dff_γ = lc_hier_st_out.loc[idx[:, "γ"], :].droplevel(level=1).reset_index().rename(
    columns={"mean": "value"}
)

merged_us_states_γ = pd.merge(
  us_states, dff_γ, left_on="STUSPS", right_on="state", how="right"
)

fig, ax = plt.subplots(1, figsize=(15, 18))
albers_epsg = 2163
ax = us_states[~us_states["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    ax=ax, linewidth=0.25, edgecolor='white', color='grey'
)

ax = merged_us_states_γ[~merged_us_states_γ["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    column='value', ax=ax, cmap='viridis', scheme="quantiles", legend=True, 
    legend_kwds={"loc": "upper center", "ncol": 3}
)
_ = ax.axis('off');


print(lc_hier_b_out[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat"]].to_latex(
  index=True, column_format="rrrrrr",
  )
)

ax = az.plot_forest(hier_data, var_names=["b"], combined=True, figsize=(10, 5))
grade_vars = [x for x in lc_hier_b_out.index if "grade" in x and not "fico" in x]
_ = ax[0].set_yticklabels(reversed(lc_hier_b_out.index.to_list()))


pooled_trace, pooled_data, _, pooled_b_out = make_az_data(omap["PR"], "pooled")
print(
  pooled_b_out[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat"]].to_latex(
     column_format="rrrrrr"
  )
)

fig = make_ppc_plot(omap["PR"])
fig.show()

hier_trace, hier_data, pr_hier_st_out, pr_hier_b_out = make_az_data(
  omap["PR"], "hier"
)
dff_η = pr_hier_st_out.loc[idx[:, "η"], "mean"].droplevel(level=1).reset_index().rename(
    columns={"mean": "value"}
)

us_states = gpd.read_file(
  "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"
)
merged_us_states_η = pd.merge(
  us_states, dff_η, left_on="STUSPS", right_on="state", how="right")

fig, ax = plt.subplots(1, figsize=(15, 18))
albers_epsg = 2163
ax = us_states[~us_states["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    ax=ax, linewidth=0.25, edgecolor='white', color='grey'
)

ax = merged_us_states_η[~merged_us_states_η["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    column='value', ax=ax, cmap='viridis', scheme="quantiles",  legend=True, 
    legend_kwds={"loc": "upper center", "ncol": 3}
)
_ = ax.axis('off');


dff_γ = pr_hier_st_out.loc[idx[:, "γ"], :].droplevel(level=1).reset_index().rename(
    columns={"mean": "value"}
)

merged_us_states_γ = pd.merge(
  us_states, dff_γ, left_on="STUSPS", right_on="state", how="right"
)

fig, ax = plt.subplots(1, figsize=(15, 18))
albers_epsg = 2163
ax = us_states[~us_states["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    ax=ax, linewidth=0.25, edgecolor='white', color='grey'
)

ax = merged_us_states_γ[~merged_us_states_γ["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    column='value', ax=ax, cmap='viridis', scheme="quantiles", legend=True, 
    legend_kwds={"loc": "upper center", "ncol": 3}
)
_ = ax.axis('off');

print(pr_hier_b_out[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat"]].to_latex(
  index=True, column_format="rrrrrr",
  )
)

ax = az.plot_forest(hier_data, var_names=["b"], combined=True, figsize=(10, 5))
grade_vars = [x for x in pr_hier_b_out.index if "grade" in x and not "fico" in x]
_ = ax[0].set_yticklabels(reversed(pr_hier_b_out.index.to_list()))


START_DATE = datetime.date(1995, 1, 1)
CRISIS_START_DATE = datetime.date(2020, 3, 14)
HOME_DIR = str(pathlib.Path.home())

with open(HOME_DIR + "/.config/gcm/gcm.toml", "rb") as f:
    config = pytoml.load(f)
    FRED_API_KEY = config["api_keys"]["fred"]

claims_az_data = claims_dict["az_data"]
claims_sum_df = claims_dict["sum_df"]
claims_trace = claims_dict["trace"]
claims_data = claims_dict["data"]
claims_epi_enc = claims_dict["epi_enc"]
claims_sim_dict = claims_dict["sim_dict"]

A = 0
κ = claims_trace["κ"]
β = claims_trace["β"]

def project_claims(state, covid_wt, sum_df, epi_enc, verbose=False):
    ''' get labor market data from STL '''
    
    def states_data(suffix, state, fred):
        ''' gets data from FRED for a list of indices '''

 
        idx = "ICSA" if state == "US" else state + suffix            
        x =  pd.Series(
                fred.get_series(
                    idx, observation_start=START_DATE), name=v
            )

        x.name = state

        return x    
    
    def forecast_claims(initval, initdate, enddate, covid_wt):
        ''' project initial claims '''
    
        μ_β = sum_df.loc["β", "mean"]
        μ_κ = sum_df.loc[["κ: COVID", "κ: Katrina"], "mean"].values
        μ_decay = covid_wt * μ_κ[0] + (1 - covid_wt) * μ_κ[1]
        
        dt_range = (
            pd.date_range(start=initdate, end=enddate, freq="W") - 
            pd.tseries.offsets.Day(1)
        )
        max_x = len(dt_range)
        
        w = np.arange(max_x)
        covid_idx = list(epi_enc.classes_).index("COVID")
        katrina_idx = list(epi_enc.classes_).index("Katrina")
        
        decay = covid_wt * κ[:, covid_idx] + (1 - covid_wt) * κ[:, katrina_idx]
        μ = np.exp(-decay * np.power(w.reshape(-1, 1), β))
        
        μ_df = pd.DataFrame(
            np.percentile(μ, q=[5, 25, 50, 75, 95], axis=1).T, 
            columns=["5th", "25th", "50th", "75th", "95th"]
        ) * initval
        μ_df["period"] = w
           
        ic = np.zeros(max_x)
        ic[0] = 1
        for j in np.arange(1, max_x, 1):
            ic[j] = np.exp(-μ_decay * np.power(j, μ_β))
        
        df = pd.concat(
            [
                pd.Series(np.arange(max_x), name="period"),
                pd.Series(ic, name="ic_ratio"),
                pd.Series(ic * initval, name="ic"),
                pd.Series((ic * initval).cumsum(), name="cum_ic")
            ], axis=1
        )
       
        df.index = dt_range
        μ_df.index = dt_range
    
        return df, μ_df
    
    fred = Fred(api_key=FRED_API_KEY)
    ic_raw = states_data("ICLAIMS", state, fred)

    init_value, init_date, last_date = (
        ic_raw[ic_raw.idxmax()], ic_raw.idxmax(), ic_raw.index[-1]
    )
    end_date  = (
        last_date + pd.tseries.offsets.QuarterEnd() + pd.tseries.offsets.DateOffset(months=3)
    )
    
    if verbose:
        print(
            f'State: {state}, {init_value}, {init_date}, {end_date}, {last_date}'
        )
    
    ic_fct, ic_pct = forecast_claims(init_value, init_date, end_date, covid_wt)
    ic_fct["state"] = state
    ic_pct["state"] = state
    
    return ic_raw, ic_fct, ic_pct, init_date, end_date

print(
  claims_sum_df[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat"]].to_latex(
     column_format="rrrrrr"
  )
)

fig, ax = plt.subplots(2, 3, figsize=(12, 6))

for i, v in enumerate(["κ: " + x for x in claims_epi_enc.classes_]):
    
    xx = claims_data[claims_data.epi_idx==i]["x"].values
    yy = claims_data[claims_data.epi_idx==i]["y"].values
    xx = xx.reshape((xx.max() + 1, 1))
    
    mu = A - κ[:, i].reshape(-1,1) * np.power(xx, β).T
    ic_hat_means = mu.mean(axis=0)
    ic_hat_se = mu.std(axis=0)
    
    j = i % 3
    l = 0 if i < 3 else 1
    ax[l, j].plot(xx, yy, 'C0.')
    ax[l, j].plot(xx, np.exp(ic_hat_means), c='k')

    ax[l, j].fill_between(
        xx[:, 0], np.exp(ic_hat_means + 1 * ic_hat_se),
        np.exp(ic_hat_means - 1 * ic_hat_se), alpha=0.6,
        color='C1'
    )
    ax[l, j].fill_between(
        xx[:, 0], np.exp(ic_hat_means + 2 * ic_hat_se),
        np.exp(ic_hat_means - 2 * ic_hat_se), alpha=0.4,
        color='C1'
    )
    ax[l, j].set_xlabel('Weeks since peak')
    ax[l, j].set_ylabel('Pct. of peak')
    ax[l, j].set_title(f'Episode: {v} = {claims_sum_df.loc[v, "mean"]}')
    
fig.tight_layout()

covid_wt = 0.9

ic_raw, fct_df, ic_pct, init_date, end_date = project_claims(
  "US", covid_wt, claims_sum_df, claims_epi_enc
)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(ic_pct["period"], ic_pct["50th"].cumsum())
ax[0].scatter(
  claims_data[claims_data.epi_idx==1]["x"], 
  (claims_data[claims_data.epi_idx==1]["y"] * ic_pct.iloc[0, 1]).cumsum()
)
ax[0].fill_between(
    ic_pct["period"], (ic_pct["25th"]).cumsum(), (ic_pct["75th"]).cumsum(), 
    alpha=0.6, color='C1'
)
ax[0].fill_between(
    ic_pct["period"], ic_pct["5th"].cumsum(), ic_pct["95th"].cumsum(), alpha=0.4,
    color='C1'
)
ax[0].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
ax[0].set(xlabel='Weeks from peak', ylabel='Cum. initial claims')

ax[1].plot(ic_pct["period"], ic_pct["50th"])
ax[1].scatter(
  claims_data[claims_data.epi_idx==1]["x"], 
  (claims_data[claims_data.epi_idx==1]["y"] * ic_pct.iloc[0, 1])
)
ax[1].fill_between(
    ic_pct["period"], (ic_pct["25th"]), (ic_pct["75th"]), alpha=0.6, color='C1'
)
ax[1].fill_between(
    ic_pct["period"], ic_pct["5th"], ic_pct["95th"], alpha=0.4,
    color='C1'
)
ax[1].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
ax[1].set(xlabel='Weeks from peak', ylabel='Initial claims')

plt.tight_layout()


def predict_hazard(originator):
    ''' generates hazard predictions '''

    horizon_date = datetime.date(2020, 9, 30)
    sub_df = make_df(hard_df_dict[originator], ASOF_DATE, horizon_date)
    
    orig_model_key = ":".join([originator, "hier"])
    aaa, zzz, s_1_df = simulate(
      sub_df, claims_dict["chg_df"], originator, ASOF_DATE, out_dict[orig_model_key]
    )

    return zzz

idx = pd.IndexSlice

lc_zzz = predict_hazard(omap["LC"])
pr_zzz = predict_hazard(omap["PR"])

fig, ax = plt.subplots(1,2, figsize=(10, 5))
for i in lc_zzz.index.get_level_values(0).to_series().sample(n=100, random_state=12345):
    lc_zzz.loc[idx[i, "2020-03-14":"2020-07-26"], ["ymean"]].reset_index().plot(
        x="edate", y="ymean", ax=ax[0], legend=False, alpha=0.25,
        title="Originator " + omap["LC"]
    )
_ = ax[0].set(xlabel='Week ending', ylabel='Hazard')

for i in pr_zzz.index.get_level_values(0).to_series().sample(n=100, random_state=12345):
    pr_zzz.loc[idx[i, "2020-03-14":"2020-07-26"], ["ymean"]].reset_index().plot(
        x="edate", y="ymean", ax=ax[1], legend=False, alpha=0.25,
        title="Originator " + omap["PR"]
    )
_ = ax[1].set(xlabel='Week ending', ylabel='Hazard')
plt.tight_layout()

zzz = pm.model_to_graphviz(out_dict[omap["PR"] + ":hier"]["model"])
_ = zzz.render(
  directory="figures",
  filename="pr_hier_model", format="png",
  cleanup=True
)

df_η = lc_hier_st_out.loc[idx[:, "η"], :].droplevel(level=1)
cnames = ["mean", "sd", "hpd_3%", "hpd_97%", "r_hat"]
c_fmt = "".join(["r"] * (1 + df_η.shape[1]))
print(df_η.to_latex(index=True))

df_η = pr_hier_st_out.loc[idx[:, "η"], :].droplevel(level=1)
print(df_η.to_latex(index=True))
