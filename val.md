---
title: "Health facility reporting rate"
weight: 3
order: 3
freeze: true
execute:
  cache: true
  warning: false
  message: false
format:
  html:
    self-contained: true
    toc: true
    toc-depth: 4
# added the following to make sure both R and python cells run, and that python cells are run in the same session and can share variables
engine: knitr
knitr:
  opts_chunk:
    python.reticulate: true
---

## Overview

In the SNT workflow, routine surveillance data are often used to calculate key indicators such as malaria incidence, test positivity rate, and confirmed case counts. To interpret and use this data properly, we need to first assess their quality and completeness. One important assessment is of what proportion of health facilities are reporting consistently and completely across geography and time.

Reporting rates provide a simple and essential metric to evaluate the completeness of routine data. They help identify where gaps in reporting may affect the reliability of indicators used in decision-making and where routine surveillance should be strengthened. Reporting rates can also be used to estimate the number of additional cases (or other count indicator) that would have also been included in routine surveillance, if all facilities had reported.

This section outlines how to calculate, inspect, and save reporting rates using a reproducible approach, while grounding all choices in dialogue with the national malaria program and SNT team.

::: {.callout-note title="Objectives" appearance="simple"}
-   Understand how to calculate reporting rates from routine health facility data
-   Visualize and interpret reporting rate patterns at any admin unit level
-   Compile and save validated reporting rate outputs for use in later SNT workflow steps
:::

## Defining and Calculating Reporting Rate for Routine Indicators

To ensure routine data can be used reliably in the SNT workflow, we need a clear and consistent method for calculating reporting rates across facilities and time. This section introduces a process to calculate monthly reporting rates for any routinely reported indicator.

### What is reporting rate?

Reporting rate is the proportion of entities, such as health facilities or community health workers, in a given admin unit that reported on an indicator during a time period of interest.

In the example on this page, we are using monthly data from DHIS2, so we calculate monthly reporting rates. However, you should calculate reporting rate for the relevant reporting period in your dataset. For example, if you are analyzing weekly surveillance data, your reporting rate should be calculated on a weekly basis.

When presenting reporting rates, it is important to specify what indicator(s) are used to define the entity as reporting. In SNT, it is best practice to re-calculate reporting rate for each indicator of interest, as reporting practice may vary across indicators within the same facility. For example, a facility may prioritize reporting confirmed cases, as their stock replenishment depends on showing consistent reporting, but neglect to report all-cause outpatient visits.

An overall reporting rate can also be calculated for each entity in a given time period. In this case, a facility may only be defined as reporting if it reports suspected cases, tested cases, confirmed cases, *and* treated cases. This aggregated reporting rate should be at most the minimum of the reporting rate of each individual indicator. 

### Establishing the denominator: Which facilities are expected to report?

Before evaluating the proportion of facilites that reported in a given reporting period, we need to first determine the number of facilities that *should* report. To avoid underestimating the reporting rate and gaining and inaccurate assessment of the quality of surveillance, the SNT team may, for example, consider:

- When calculating reporting rate for confirmed malaria cases, exclude facility types that do not test or treat malaria. For example, HIV clinics or maternity wards. 

- When calculating reporting rate for malaria admissions, exclude facility types that only handle outpatients, for example community health workers or health posts.

- When calculating reporting rate for an indicator, exclude facilities that have closed, that are not yet active, or are temporarily nonfunctional. This avoids penalizing newly opened facilities that weren’t expected to report in earlier months, or facilities that are permanently or temporarily closed and therefore are not expected to report.

Up-to-date master facility lists (MFL) that track facility type and activity status are very helpful for determining which health facilities should be included in the denominator for reporting rate of each indicator, for each reporting period. In the absence of an MFL, or official determination of activity status, it is still possible to infer which health facilities should be excluded. 

For more guidance on how to consider active vs inactive facilities, please see the code library page [Determining active and inactive status](https://ahadi-analytics.github.io/snt-code-library/english/library/data/health_facilities/active_status.html).

Code for excluding by specific facility type is included on this page.

::: {.callout-important title="Consult the SNT team"}

Consult the SNT team to understand how to determine which facilities, if any, should not be included in the denominator for reporting rate. National practices vary, and the surveillance focal person on the SNT team should explain what would be appropriate for each indicator.

:::

### Calculating reporting rates

Once health facility activity status has been established, reporting rates can be calculated. These rates reflect the proportion of *expected* facilities that submitted valid data for a given indicator in a given time period.

For each indicator of interest, reporting rate is defined as:

$$
\text{Indicator Reporting Rate}_{a,t} =
\frac{o_{a,t}}{e_{a,t}}
$$

Where:

- $a$ is the administrative unit (e.g. chiefdom or district)
- $t$ is the time period (e.g. “2022-03”)
- $o_{a,t}$ is the number of **observed** facilities in unit $a$ during time $t$
- $e_{a,t}$ is the number of **expected** facilities in unit $a$ during time $t$

Observed facilities are those that submitted a valid (non-missing) value for the indicator of interest during time $t$.

Expected facilities are those who were expected to report during time $t$ (see previous section). Remember to consult with the SNT team to decide what rules determine whether a facility is expected to report.

#### Worked example

**PLEASE MAKE SURE THIS EXAMPLE IS USING ACTUAL NUMBERS FOR KAILAHUN DISTRICT IN MARCH 2022**

Suppose we are calculating the reporting rate for total confirmed cases for **Kailahun District** in March 2022.

- There are 6 health facilities in Kailahun that have ever submitted data on any key malaria indicator
- All 6 submitted their first report **on or before March 2022**, so they are assumed to be active and **expected to report** that month
- Of these, 4 facilities reported a valid value for `conf` (total confirmed cases) in March 2022 → 4 are **observed reporting**
- The other two do not have a valid value for total confirmed cases (they show `NA` in the database) for March 2022

The reporting rate is calculated as:

$$
\text{Reporting Rate for Total Confirmed Cases}_{\text{Kailahun}, \text{Mar 2022}} = \frac{4}{6} = 0.67
$$

### Weighted reporting rates

For some SNT applications, a **weighted** reporting rate may be of interest. The weighted reporting rate is an estimate of the proportion of an indicator's expected total counts in a given admin unit over a given time period that was reported into routine surveillance. 

This means that if a non-reporting facility generally reports *fewer* confirmed cases than average in its admin unit, the weighted reporting rate for confirmed cases is *higher* than the unweighted reporting rate (fewer cases are missing). Conversely, if a non-reporting facility generally reports relatively *many* confirmed cases for its admin unit, the weighted reporting rate for confirmed cases is *lower* than the unweighted reporting rate (more cases are missing).

#### Calculating weighted reporting rates

**PLACEHOLDER EXPLANATION BELOW. WOULD BE GREAT TO INCLUDE THE SIMPLE EXAMPLE FROM BEA'S SLIDE DECK incl the illustration if possible**

The monthly weighted reporting rate for each health district was determined as follows. For each calendar month (January through December), health facility weights were calculated by dividing the health facility’s average number of malaria cases reported for that month, across all years of data, by the district sum of the average number of malaria cases reported for that month, across all years of data. For dates in which a health facility is inactive, the average number of malaria cases reported for that month-and-year pair would be 0, and the weights for the health facilities in that district for that date would be calculated as usual. The district monthly weighted reporting rate was then calculated by summing the weights of the active health facilities. This value captures the proportion of expected confirmed malaria cases that are reported at the district level each month by the active health facilities included in the HMIS database. 

#### How do I know whether to use unweighted or weighted reporting rate?

To assess the performance of the routine surveillance system, the unweighted reporting rate is likely to be more appropriate.

To estimate an admin unit's unreported confirmed cases (as an example indicator), the weighted reporting rate may be the more accurate option, as it will account for the size of the non-reporting facility. If you choose to use weighted reporting rates, best practice is to also calculate the unweighted reporting rates for the same indicator, compare the two outputs, and discuss with the SNT team.

**WOULD BE GREAT TO INCLUDE IN THE STEP BY STEP EXAMPLE VISUALIZATIONS. BOTH SEBASTIAN'S LINE PLOT VERSION AND OUSMANE'S HEATMAPS**

::: {.callout-important title="Consult the SNT team"}

If you think the weighted reporting rate might be the better option, produce reporting rates for using both methods and present to the SNT team. Discuss with the SNT team to understand how, where, when, and why the two reporting rates are different. Together with the SNT team, discuss which to use for downstream analysis.

:::


## Step-by-Step

***VT: in the example below I am using the current pre-processed data file I was given, which I had to adapt in the active/inactive page***
***I will update to the reference pre-processed data once we agree on the import page with the team***

Now that we've defined how reporting rates are constructed—by identifying active facilities and calculating observed reporting—we move into the step-by-step process for implementing this in code using example DHIS2 data from Sierra Leone.  In this section, we walk through the steps for calculating and visualising monthly reporting rates. Each step is designed to guide you through the process. Follow the notes in the code, especially where edits are required.

To skip the step-by-step explanation, jump to the full code at the end of this page.

<a href="#bottom-section"> <button class="btn btn-primary">Jump to Full Code</button> </a>

::: {.callout-note title="Objectives" appearance="\"simple"}
- Calculate monthly reporting rates
- Visualise reporting rate over time, by indicator
:::

### Step 1: Import relevant packages

In this step, we load the necessary packages to run this section.

::: panel-tabset

## R

```{r}

```

## Python
```{python}
#| eval: true
#| echo: true

import pandas as pd
import numpy as np
from pyhere import here
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import matplotlib.patches as mpatches
```

:::

### Step 2: Import Data

Now we import the DHIS2 dataset that was initially processed in the [DHIS2 Data Preprocessing](https://ahadi-analytics.github.io/snt-code-library/english/library/data/routine_cases/import.html) section of this code library.

::: panel-tabset

## R

```{r}

```

## Python

```{python}
#| eval: false
# import pre-processed data
df0 = pd.read_csv(here("english/data_r/routine_cases/df0_temp_VT.csv"))

# check head of imported data
df0.head(10).style
```

```{python}
#| echo: false
#| eval: true
#| output: true

# VAL, I HAVE THIS CHUNK AS IT DOES THE ACTUAK IMPORTING OF DATA
# THE ABOVE IS TO JUST A DUMMY WHICH DOESNT RUN BUT MAKES IT LOOK LIEK WE ARE
# IMPORTING FROM THE ABOVE PATH. THIS IS TO STAY CONSISTENT THROUGH OUT THE CODE LIBRARY
# USING OUR BEST PRACTICES IN THE FOR ANALYST SECTIONS OF THE STARTER PAGES.

# VT: SOUNDS GOOD, MAYBE FEELS A LITTLE REPETITIVE IN THIS CASE AS IT IS EXCATLY THE SAME CODE IN BOTH CELLS? WE CAN DISCUSS ON WEDNESDAY
# I CHANGED TO USE PANDAS INSTEAD OF PYREADR - PYHTON USERS LIKELY WON'T USE PYREADR

# import pre-processed data
df0 = pd.read_csv(here("english/data_r/routine_cases/df0_temp_VT.csv"))

# check head of imported data
df0.head(10).style
```
:::

### Step 3: Import your denominator dataframe - *dfden*

Here we import the reporting rate denominator dataframe we built in section **link to active status section**.

::: {.panel-tabset}

## R

```{r}
#| message: false
#| warning: false
#| code-fold: true
#| code-summary: Show the code
```

## Python

```{python}
#| message: false
#| warning: false
#| eval: true
#| output: true
#| code-fold: true
#| code-summary: Show the code

# import reporting rate denominator data
dfden = pd.read_csv(here("english/data_r/routine_cases/dfden.csv"))

# inspect
dfden.head(10).style
```
:::

### Step 4: Define function to calculate reporting rate

Now we define a function to calculate the monthly reporting rate, for a given indicator, at a given admin unit level.

::: panel-tabset

## R

```{r}

```

## Python

```{python}
#| eval: true
#| echo: true

# VT: REMEMBER TO CHANGE TO _uid WHEN AVAILABLE FROM import.qmd AFTER DISCUSSING WITH TEAM
def compute_RR(base_df, dfden, level, indicator):
    cols_group = ['YM', f'adm{level}']
    df = base_df.copy()
    
    # count number of non-null reports for confirmed cases
    df = df.groupby(cols_group)[indicator].count().reset_index()
    
    # aggregate denominator at this level
    temp = (dfden.groupby(cols_group)['denominator']
            .sum(min_count = 1)
            .reset_index())
    
    # merge numerator and denominator
    df = df.merge(temp, on = cols_group, how = 'outer', validate = '1:1')
    
    # compute reporting rate
    df.insert(len(df.columns), f'{indicator}_RR', df[indicator].div(df['denominator']))
    df = df[[f'adm{level}', 'YM', f'{indicator}_RR']]
    
    return df
```
:::

### Step 5: Calculate reporting rate for a given indicator, at a given admin unit level

::: {.panel-tabset}

## R

```{r}
#| message: false
#| warning: false
#| code-fold: true
#| code-summary: Show the code
```

## Python

```{python}
#| message: false
#| warning: false
#| eval: true
#| code-fold: true
#| code-summary: Show the code

# set the indicator and admin unit level we want to calculate reporting rate at
level = 3
indicator = 'conf'

# use our function to calcupate the monthly reporting rate, in this case for the indicator `conf`, at admin level 3
df = compute_RR(df0, dfden, level, indicator)

# save results
df.to_csv(here(f"english/data_r/routine_cases/Monthly_adm{level}_RR_{indicator}.csv"), index = None)

# inspect results
df.head(10).style
```
:::

### Step 6: Visualise indicator-specific reporting rates

***VT: Do we want additional visualisations here, line plots over time maybe, maps? To discuss with team***

#### Step 6.1 Multiple indicators, nationally

::: {.panel-tabset}

## R

```{r}
#| message: false
#| warning: false
#| code-fold: true
#| code-summary: Show the code
```

## Python

```{python}
#| message: false
#| warning: false
#| code-fold: true
#| code-summary: Show the code

# Make a heatmap of monthly reporting rate by variable
indicators =  ['allout', 'test', 'conf', 'pres', 'maltreat']

df = pd.DataFrame(columns = ['YM'])
for i in indicators:
    t = compute_RR(df0, dfden, 0, i).drop('adm0', axis = 1)
    # make RR a percentage for visual
    t[f'{i}_RR'] = t[f'{i}_RR']*100
    df = df.merge(t, on = 'YM', how = 'outer', validate = '1:1')

# rearrange for heatmap
df = df.set_index('YM').T

fig, ax = plt.subplots(figsize = (15, 4))
cbar_ax = fig.add_axes([.91, .2, .02, .6])

sns.heatmap(ax = ax
            , data = df
            , cmap = 'viridis'
            , cbar_ax = cbar_ax
            , vmin = 0
            , vmax = 100
            , cbar_kws = {'label': '%'})

ax.set_xlabel('')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([l.get_text()[0:-3] for l in ax.get_yticklabels()], rotation = 0, ha = 'right')

ax.set_title('Monthly reporting rate')

# Save
# discuss with team
```
:::

#### Step 6.2 Single indicator, by admin unit

::: {.panel-tabset}

## R

```{r}
#| message: false
#| warning: false
#| code-fold: true
#| code-summary: Show the code
```

## Python

```{python}
#| message: false
#| warning: false
#| code-fold: true
#| code-summary: Show the code

# VT to clean this up after agreeing on import.qmd with team

dftree = df0[['adm0', 'adm1', 'adm2', 'adm3']].drop_duplicates().reset_index(drop = True)

level = 3
level_yaxis = 1
indicator = 'conf'
fs = 15

df = compute_RR(df0, dfden, level, indicator)
df[f'{indicator}_RR'] = df[f'{indicator}_RR']*100

# rearrange for heatmap
df = df.pivot(index = [f'adm{level}'], columns = 'YM', values = f'{indicator}_RR')
t = dftree[[f'adm{level_yaxis}', f'adm{level}']].drop_duplicates()
df = df.merge(t, on = f'adm{level}', how = 'left', validate = 'm:1')

df = (df.sort_values(by = [f'adm{level_yaxis}', f'adm{level}'])
      .reset_index(drop= True)
      .set_index([f'adm{level_yaxis}', f'adm{level}']))


fig, ax = plt.subplots(figsize = (30, 15))
cbar_ax = fig.add_axes([.91, .2, .02, .6])

sns.heatmap(ax = ax
            , data = df
            , cmap = 'viridis'
            , cbar_ax = cbar_ax
            , vmin = 0
            , vmax = 100
            , cbar_kws = {'label': '%'})


ax.set_xlabel('')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

# trick to label adm unit names nicely on the yaxis
yticklabels = [l.get_text().split('-')[0] for l in ax.get_yticklabels()]
t = pd.DataFrame(yticklabels).reset_index()
t1 = t.groupby(0)['index'].mean().astype(int).reset_index()
t.insert(0, 'pos', t[0].map(dict(zip(t1[0], t1['index']))))
t = t[[0, 'pos', 'index']]
t[0] = np.where(t.pos == t['index'], t[0], '')
test = t[0].to_list()

ax.set_yticks(ax.get_yticks())
ax.set_yticklabels(test, size = fs)
ax.set_ylabel(f'adm{level}', size = fs)

ylabel_mapping = OrderedDict()

for adm1, adm3_uid in df.index:
    ylabel_mapping.setdefault(adm1, [])
    ylabel_mapping[adm1].append(adm3_uid)

hline = []
new_ylabels = []

for adm1, adm3_list in ylabel_mapping.items():
    adm3_list[0] = "{} - {}".format(adm1, adm3_list[0])
    new_ylabels.extend(adm3_list)

    if hline:
        hline.append(len(adm3_list) + hline[-1])
    else:
        hline.append(len(adm3_list))

ax.hlines(hline, xmin = -10, xmax = 0, color = "grey", linewidth = 2, clip_on = False)

# color NaN values
colour = 'grey'
ax.set_facecolor(colour)
handle = [mpatches.Patch(color = colour, label = 'No data')]
ax.legend(handles = handle
          , fontsize = fs
      , bbox_to_anchor = (1, 0)
      , loc = 'lower left')

ax.set_title(f'Monthly {indicator} reporting rate by adm{level}', size = fs)
```
:::

<!-- ### Step X: template to copy paste for new blocks

::: {.panel-tabset}

## R

```{r}
#| message: false
#| warning: false
#| code-fold: true
#| code-summary: Show the code
```

## Python

```{python}
#| message: false
#| warning: false
#| code-fold: true
#| code-summary: Show the code
```

::: -->

<div id="bottom-section"></div>

## Full code

<b>Find the full code script for calculating reporting rates below.</b>

::: {.panel-tabset}

## R
```{r}
#| warning: false
#| message: false
#| package: false
#| eval: false
#| code-fold: true
#| code-summary: Show full code
```

## Python

:::
