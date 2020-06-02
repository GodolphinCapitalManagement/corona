
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

import joblib
import numpy as np
import pandas as pd

import feather

import pymc3 as pm
import arviz as az

from scipy.special import expit

from analytics import utils

files_dir = "/home/gsinha/admin/db/dev/Python/projects/scripts/misc/"
sys.path.append(files_dir)
import common

pymc3_dict = {}
data_scaler_dict = {}
risk_scaler_dict = {}

data_dict = {}
hard_dict = {}
risk_dict = {}

X_dict = {}
X_risk_dict = {}

asof_date_dict = {}
ic_date_dict = {}

cat_vars_dict = {}
cat_var_names_dict = {}

for i in ["LC", "PR"]:
  fname = files_dir + "hierarchical_r_" + i + ".pkl"
  with open(fname, "rb") as f:
    pymc3_dict[i] = joblib.load(f)

    data_scaler_dict[i] = pymc3_dict[i]["data_scaler"]
    risk_scaler_dict[i] = pymc3_dict[i]["risk_scaler"]
    data_dict[i] = pymc3_dict[i]["data_df"]
    hard_dict[i] = pymc3_dict[i]["hard_df"]
    risk_dict[i] = pymc3_dict[i]["risk_df"]
    X_dict[i] = pymc3_dict[i]["X"]
    X_risk_dict[i] = pymc3_dict[i]["X_risk"]

    # PARAMS
    asof_date_dict[i] =  pymc3_dict[i]["asof_date"]
    ic_date_dict[i] =  pymc3_dict[i]["ic_date"]

    cat_vars_dict[i] = pymc3_dict[i]["cat_vars"]
    cat_var_names_dict[i] = pymc3_dict[i]["cat_var_names"]

with open(files_dir + "risk_df.feather", "rb") as f:
  risk_df = feather.read_dataframe(f)

def make_az_data(originator, pymc3_dict, X, X_risk, risk_df):
  ''' make az data instance for originator '''

  hier_model = pymc3_dict[originator]["model"]
  hier_trace = pymc3_dict[originator]["trace"]

  with hier_model:
    ppc = pm.sample_posterior_predictive(hier_trace, progressbar=False)

  az_data = az.from_pymc3(
      trace=hier_trace,
      posterior_predictive=ppc,
      model=hier_model,
      coords={
          'covars': X.columns.to_list(),
          'states': risk_df.state.to_list(),
          'beta': X_risk.columns.to_list(),
          "Δ": ["Δ_" + x for x in risk_df.state.to_list()]
      },
      dims={
        'a': ["states"], "α": ["states"], "β": ["beta"], 'Δ_a': ["Δ"],
        'b': ['covars']
      }
  )

  return az_data, hier_model, hier_trace, ppc

df = []
for i in ["PR", "LC"]:
    df.append(data_dict[i])
hard_df = pd.concat(df, sort=False, ignore_index=True)

def wavg(x):
    return np.average(
        x, weights=hard_df.loc[x.index, "original_balance"]
    )

aaa = hard_df.groupby(["originator", "grade"]).agg(
    n=('loan_id', "count"),
    original_balance=('original_balance', sum),
    wac=('original_rate', wavg),
    age=('age', wavg),
    fico=('fico', wavg),
    term=('original_term', wavg),
    defer=('defer', np.mean)
)

bbb = hard_df.groupby(["originator"]).agg(
    n=('loan_id', "count"),
    original_balance=('original_balance', sum),
    wac=('original_rate', wavg),
    age=('age', wavg),
    fico=('fico', wavg),
    term=('original_term', wavg),
    defer=('defer', np.mean)
)

bbb.index = pd.MultiIndex.from_tuples(
    [('LC', 'ALL'), ('PR', 'ALL')], names=['originator', 'grade']
)

aaa = pd.concat([aaa, bbb])

ccc = pd.concat(
    [
        pd.Series(hard_df["loan_id"].apply("count"), name="n"),
        pd.Series(hard_df["original_balance"].sum(), name="original_balance"),
        hard_df[["original_rate", "age", "fico", "original_term"]].apply(wavg).to_frame().T.rename(
            columns={"original_term": "term", "original_rate": "wac"}),
        pd.Series(hard_df["defer"].mean(), name="defer")
    ], axis=1
)
ccc.index = [('ALL', 'ALL')]

ddd = pd.concat([aaa, ccc])
ddd["pct"] = ddd["original_balance"]/ddd.loc[pd.IndexSlice["ALL", "ALL"],  "original_balance"]
ddd.index.names = ["Originator", "Grade"]
cfmt = "".join(["r"] * (ddd.shape[1] + 2))
print(
    ddd.to_latex(
      index=True, multirow=True, 
      header=["N", "Balance", "WAC", "WALA", "FICO", "WAOT", "Defer", "Share"],
      bold_rows=True,
      formatters={
        "original_balance": utils.dollar,
        "n": utils.integer, "fico": utils.number,
        "term": utils.number, "age": utils.number,
        "pct": utils.percent, "defer": utils.percent,
        "wac": utils.percent
      },
      column_format=cfmt,
      multicolumn_format="r",
    ))

one_line = ddd.loc[pd.IndexSlice["ALL", :], :]

purpose_tbl = common.summary_by_group(
    ["originator", "purpose"], hard_df
)
purpose_tbl.index.names = ["Originator", "Purpose"]
cfmt = "".join(["r"] * (purpose_tbl.shape[1] + 2))
print(
    purpose_tbl.to_latex(
      index=True, multirow=True, 
      header=["N", "Balance", "WAC", "WALA", "FICO", "WAM", "Defer", "Share"],
      bold_rows=True,
      formatters={
        "original_balance": utils.dollar,
        "n": utils.integer, "fico": utils.number,
        "term": utils.number, "age": utils.number,
        "pct": utils.percent, "defer": utils.percent,
        "wac": utils.percent
      },
      column_format=cfmt,
      multicolumn_format="r",
    ))

emp_tbl = common.summary_by_group(
    ["originator", "employment_status"], hard_df
)
emp_tbl = emp_tbl.fillna(0)
emp_tbl.index.names = ["Originator", "Employment"]
cfmt = "".join(["r"] * (emp_tbl.shape[1] + 2))
print(
    emp_tbl.to_latex(
      index=True, multirow=True, 
      header=["N", "Balance", "WAC", "WALA", "FICO", "WAM", "Defer", "Share"],
      bold_rows=True,
      formatters={
        "original_balance": utils.dollar,
        "n": utils.integer, "fico": utils.number,
        "term": utils.number, "age": utils.number,
        "pct": utils.percent, "defer": utils.percent,
        "wac": utils.percent
      },
      column_format=cfmt,
      multicolumn_format="r",
    ))

homeowner_tbl = common.summary_by_group(
    ["originator", "home_ownership"], hard_df
)
homeowner_tbl.index.names = ["Originator", "Homeownership"]
cfmt = "".join(["r"] * (homeowner_tbl.shape[1] + 2))
print(
    homeowner_tbl.to_latex(
      index=True, multirow=True, 
      header=["N", "Balance", "WAC", "WALA", "FICO", "WAM", "Defer", "Share"],
      bold_rows=True,
      formatters={
        "original_balance": utils.dollar,
        "n": utils.integer, "fico": utils.number,
        "term": utils.number, "age": utils.number,
        "pct": utils.percent, "defer": utils.percent,
        "wac": utils.percent
      },
      column_format=cfmt,
      multicolumn_format="r",
    ))

term_tbl = common.summary_by_group(
    ["originator", "original_term"], hard_df
)
term_tbl.index.names = ["Originator", "Term"]
cfmt = "".join(["r"] * (term_tbl.shape[1] + 2))
print(
    term_tbl.to_latex(
      index=True, multirow=True, 
      header=["N", "Balance", "WAC", "WALA", "FICO", "WAM", "Defer", "Share"],
      bold_rows=True,
      formatters={
        "original_balance": utils.dollar,
        "n": utils.integer, "fico": utils.number,
        "term": utils.number, "age": utils.number,
        "pct": utils.percent, "defer": utils.percent,
        "wac": utils.percent
      },
      column_format=cfmt,
      multicolumn_format="r",
    ))

dq_tbl = common.summary_by_group(
    ["originator", "dq_grp"], hard_df
)
dq_tbl.index.names = ["Originator", "DQ Status"]
cfmt = "".join(["r"] * (dq_tbl.shape[1] + 2))
print(
    dq_tbl.to_latex(
      index=True, multirow=True, 
      header=["N", "Balance", "WAC", "WALA", "FICO", "WAM", "Defer", "Share"],
      bold_rows=True,
      formatters={
        "original_balance": utils.dollar,
        "n": utils.integer, "fico": utils.number,
        "term": utils.number, "age": utils.number,
        "pct": utils.percent, "defer": utils.percent,
        "wac": utils.percent
      },
      column_format=cfmt,
      multicolumn_format="r",
    ))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.barplot(data=risk_dict["LC"], x="state", y="pct_low_risk_s", ax=ax)
ax.set_ylabel("Low-risk employment share: Z-Score")
ax.set_xlabel("State")
plt.xticks(rotation=45, fontsize=8)
_ = sns.despine()

zzz = pm.model_to_graphviz(pymc3_dict["PR"]["model"])
_ = zzz.render(
  directory="figures",
  filename="pr_hier_model", format="png",
  cleanup=True
)

def create_state_aggs(originator, hard_df, risk_df):
    ''' pct deferment vs pct low risk '''
    
    xbar_df = hard_df.groupby(["originator", "state"]).agg(
        n=("loan_id", "count"),
        k=('defer', np.sum),
        pct=('defer', np.mean),
        balance=('cur_note_amount', sum)
    ).loc[pd.IndexSlice[originator, :], :].droplevel(0).reset_index()
    
    xbar_df = pd.merge(xbar_df, risk_df, on="state")
    
    return xbar_df

def defer_prob(df, hier_trace):
    ''' deferment plots actual vs hierarchical '''
    
    data_df = df.copy()
    
    data_df = pd.concat(
        [
            data_df,
            pd.Series(hier_trace["xbeta"].mean(axis=0), name="μ_xbeta"),
            pd.Series(hier_trace["xbeta"].std(axis=0), name="σ_xbeta")
        ], axis=1
    )
    
    data_df["h_mean"] = common.invcloglog(data_df["μ_xbeta"])
    data_df["h_lo"] = common.invcloglog(data_df["μ_xbeta"] - data_df["σ_xbeta"])
    data_df["h_hi"] = common.invcloglog(data_df["μ_xbeta"] + data_df["σ_xbeta"])
    
    F_df = pd.merge(
        (1 - np.exp(-data_df.groupby("loan_id")[["h_mean"]].sum())).rename(columns={"h_mean": "F_mean"}).reset_index(),
        (1 - np.exp(-data_df.groupby("loan_id")[["h_lo"]].sum())).rename(columns={"h_lo": "F_lo"}).reset_index(),
        on="loan_id"
    ).merge(
        (1 - np.exp(-data_df.groupby("loan_id")[["h_hi"]].sum())).rename(columns={"h_hi": "F_hi"}).reset_index(),
        on="loan_id"
    )
    
    data_df = pd.merge(data_df, F_df, on="loan_id", how="left")
    data_df.drop_duplicates(subset=["loan_id"], keep="last", inplace=True)
    data_df = data_df.groupby("state").agg(
        F_mean=('F_mean', np.mean),
        F_lo=('F_lo', np.mean),
        F_hi=('F_hi', np.mean)
    ).reset_index()
        
    return data_df

#
originator = "LC"
X = X_dict[originator]
X_risk = X_risk_dict[originator]
risk_df = risk_dict[originator]
data_df = data_dict[originator]

az_data, hier_model, hier_trace, ppc = make_az_data(
  originator, pymc3_dict, X, X_risk, risk_df
)


_, ax = plt.subplots(figsize=(10, 5))
y_hat = np.array([x.mean() for x in ppc['yobs']])

ax.hist(y_hat, bins=19, alpha=0.5)
ax.axvline(data_df["defer"].mean())
ax.set(xlabel='Deferment Pct.', ylabel='Frequency')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

pctile = np.percentile([x.mean() for x in ppc['yobs']], q=[5, 95])
_ = ax.text(0.016, 1100, f'95% HPD: [{pctile[0]:.2%}, {pctile[1]:.2%}]')

β_out = az.summary(az_data, var_names=['β'],round_to=3)
β_out.index = X_risk.columns.to_list()
print(β_out[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat"]].to_latex(
  index=True, column_format="rrrrrr"
  )
)


state_agg_df = create_state_aggs(originator, hard_df, risk_df)
defer_df = defer_prob(data_df, hier_trace)

xbar_df = pd.merge(state_agg_df, defer_df, on="state")
xbar_df.sort_values(by=["pct"], inplace=True)
print(state_agg_df.tail())

# plot 
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
x = np.arange(xbar_df.index.max()+1)

ax.plot(x, xbar_df.F_mean, color="green", label="Bayes-shrinkage survivor", linestyle="--")
ax.scatter(x, xbar_df.pct, color="red", label="Naive MLE", alpha=0.5)
ax.fill_between(
    x, xbar_df.F_lo, xbar_df.F_hi, alpha=0.25, color='C1'
)

ax.set_xlabel("State")
ax.set_ylabel(f'Deferment Pct.: Week {data_df["stop"].astype(int).max()-1}')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(x)
ax.set_xticklabels(xbar_df.state, rotation=45, size=12)
plt.legend(loc="upper left")
sns.despine(left=True)

b_out = az.summary(az_data, var_names=['b'], round_to=3)
b_out["odds"] = np.exp(b_out["mean"])
b_out.index = X.columns.to_list()
print(b_out[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat", "odds"]].to_latex(
  index=True, column_format="rrrrrrr", formatters={
    "odds": utils.number
  },
  )
)

ax = az.plot_forest(az_data, var_names=["b"], combined=True, figsize=(10, 5))
_ = ax[0].set_yticklabels(reversed(X.columns.to_list()))

originator = "PR"
X = X_dict[originator]
X_risk = X_risk_dict[originator]
risk_df = risk_dict[originator]
data_df = data_dict[originator]

az_data, hier_model, hier_trace, ppc = make_az_data(
  originator, pymc3_dict, X, X_risk, risk_df
)

_, ax = plt.subplots(figsize=(10, 5))
y_hat = np.array([x.mean() for x in ppc['yobs']])

ax.hist(y_hat, bins=19, alpha=0.5)
ax.axvline(data_df["defer"].mean())
ax.set(xlabel='Deferment Pct.', ylabel='Frequency')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

pctile = np.percentile([x.mean() for x in ppc['yobs']], q=[5, 95])
_ = ax.text(0.016, 1100, f'95% HPD: [{pctile[0]:.2%}, {pctile[1]:.2%}]')

β_out = az.summary(az_data, var_names=['β'])
β_out.index = X_risk.columns.to_list()
print(β_out[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat"]].to_latex(
  index=True, column_format="rrrrrr"
  )
)

state_agg_df = create_state_aggs(originator, hard_df, risk_df)
defer_df = defer_prob(data_df, hier_trace)
xbar_df = pd.merge(state_agg_df, defer_df, on="state")
xbar_df.sort_values(by=["pct"], inplace=True)

# plot 
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
x = np.arange(xbar_df.index.max()+1)

ax.plot(x, xbar_df.F_mean, color="green", label="Bayes-shrinkage survivor", linestyle="--")
ax.scatter(x, xbar_df.pct, color="red", label="Naive MLE", alpha=0.5)
ax.fill_between(
    x, xbar_df.F_lo, xbar_df.F_hi, alpha=0.25, color='C1'
)

ax.set_xlabel("State")
ax.set_ylabel(f'Deferment Pct.: Week {data_df["stop"].astype(int).max()-1}')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(x)
ax.set_xticklabels(xbar_df.state, rotation=45, size=12)
plt.legend(loc="upper left")
sns.despine(left=True)

b_out = az.summary(az_data, var_names=['b'], round_to=3)
b_out["odds"] = np.exp(b_out["mean"])
b_out.index = X.columns.to_list()
print(b_out[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat", "odds"]].to_latex(
  index=True, column_format="rrrrrrr", formatters={
    "odds": utils.number
    }
  )
)

ax = az.plot_forest(az_data, var_names=["b"], combined=True, figsize=(10, 5))
_ = ax[0].set_yticklabels(reversed(X.columns.to_list()))

fname = files_dir + "claims.pkl"
with open(fname, "rb") as f:
  claims_dict = joblib.load(f)

claims_az_data = claims_dict["az_data"]
claims_sum_df = claims_dict["sum_df"]
claims_trace = claims_dict["trace"]
claims_data = claims_dict["data"]
claims_epi_enc = claims_dict["epi_enc"]
claims_sim_dict = claims_dict["sim_dict"]

A = 0
κ = claims_trace["κ"]
β = claims_trace["β"]

def project_claims(state, covid_wt, sum_df, epi_enc, max_x=13, verbose=False):
    ''' get labor market data from STL '''
    
    def states_data(suffix, state, fred):
        ''' gets data from FRED for a list of indices '''

        idx = "ICSA" if state == "US" else state + suffix            
        x =  pd.Series(
                fred.get_series(
                    idx, observation_start=common.START_DATE), name=v
            )

        x.name = state

        return x
    
    def forecast_claims(initval, initdate, max_x, covit_wt):
        ''' project initial claims '''
    
        μ_β = sum_df.loc["β", "mean"]
        μ_κ = sum_df.loc[["κ: COVID", "κ: Katrina"], "mean"].values
        μ_decay = covid_wt * μ_κ[0] + (1 - covid_wt) * μ_κ[1]
        
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
        dt_range = pd.date_range(initdate, periods=max_x, freq="W") - pd.Timedelta(days=1)
        df.index = dt_range
        μ_df.index = dt_range
    
        return df, μ_df
    
        
    fred = common.Fred(api_key=common.FRED_API_KEY)
    ic_raw = states_data("ICLAIMS", state, fred)

    init_value, init_date = ic_raw[ic_raw.idxmax()], ic_raw.idxmax()
    if verbose:
      print(f'State: {state}, {init_value}, {init_date}')
    
    ic_fct, ic_pct = forecast_claims(init_value, init_date, max_x, covid_wt)
    ic_fct["state"] = state
    ic_pct["state"] = state
    
    return ic_raw, ic_fct, ic_pct, init_date

print(
  claims_sum_df[["mean", "sd", "hpd_3%", "hpd_97%", "r_hat"]].to_latex(
    bold_rows=True, column_format="rrrrrr"
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

ic_raw, fct_df, ic_pct, init_date = project_claims(
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
