## Python

The code library is designed to work seamlessly with Python-based workflows. All examples assume you are working in a modern Python environment such as Jupyter Lab, VS Code with the Python extension, or PyCharm, which offer reliable environments for executing code cells, managing file paths, working with notebooks, and integrating version control via Git. 

At minimum, you should have:
- Python version 3.8 or higher (download from [python.org](https://www.python.org/downloads/))
- A code editor or IDE (we recommend [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) or [VS Code](https://code.visualstudio.com/))
- An active internet connection for installing packages and downloading external data when needed

All package management is handled using standard Python approaches. You'll see this pattern used throughout the code library:

```python
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
```

Many of the workflows rely on pre-built functions from the [`sntutils`](https://github.com/ahadi-analytics/sntutils) Python package which handles common tasks like downloading, cleaning, aggregating, or visualizing data. Your main task is to supply the right inputs and understand how the output fits into the broader pipeline.

> If you encounter issues with a function in sntutils and the built-in help or documentation doesn't resolve it, please contact [`info@appliedhealthanalytics.org`](info@appliedhealthanalytics.org) for support. This ensures that any bugs or unclear behaviors are flagged and addressed centrally.

For example, the function below downloads monthly CHIRPS rainfall rasters for January to March 2022 across Africa and saves them in a local folder:

```python
from sntutils.climate import download_chirps

# Download Africa monthly rainfall for Jan to Mar 2022
download_chirps(
    dataset="africa_monthly",
    start="2022-01", 
    end="2022-03",
    out_dir="data/chirps"
)
```

You're not expected to look inside these functions or modify them—they're built to simplify the workflow and reduce errors.

---

## Python

Here are some recommended resources for getting started with Python, all of which are free and well-regarded in the data science and public health communities. These will help you build the baseline skills needed to work with the SNT code library effectively:

**Beginner books and tutorials**

- [Python for Data Analysis (3rd Edition)](https://wesmckinney.com/book/) by Wes McKinney: An excellent introduction to Python data analysis using pandas, numpy, and matplotlib. Covers data import, cleaning, transformation, and visualization with practical examples.

- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) by Al Sweigart: Free online book that teaches Python fundamentals through practical automation tasks. Great for building confidence with basic syntax and file operations.

**Self-paced interactive tutorials**

- [Kaggle Learn - Python](https://www.kaggle.com/learn/python): Free, browser-based interactive courses covering Python fundamentals, pandas, data visualization, and machine learning. Includes hands-on exercises with real datasets.

- [DataCamp: Introduction to Python](https://www.datacamp.com/courses/intro-to-python-for-data-science): Interactive course covering Python basics, lists, functions, and packages. Includes browser-based coding challenges (free tier with limited access).

**Videos and MOOCs**

- [Python for Everybody Specialization (Coursera)](https://www.coursera.org/specializations/python): University of Michigan course series covering Python programming fundamentals through data analysis. Free to audit.

- [CS50's Introduction to Programming with Python (Harvard)](https://cs50.harvard.edu/python/2022/): Comprehensive introduction to programming concepts using Python. Free and includes problem sets.

**Other useful resources**

- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf): Quick reference for pandas operations including data import, selection, grouping, and merging.

- [Real Python](https://realpython.com/): High-quality tutorials and articles covering Python fundamentals through advanced topics. Many free tutorials available.

- [Python Package Index (PyPI)](https://pypi.org/): Repository for finding and learning about Python packages used in data analysis.

---

## Python

**General Style Principles**

- *Follow PEP 8:* Python has an official style guide (PEP 8) that promotes readable, consistent code. This includes using snake_case for variable names, limiting lines to 79-88 characters, and using meaningful variable names.

- *Use explicit imports:* Import specific functions rather than using wildcards. Write `from pandas import DataFrame` or `import pandas as pd` rather than `from pandas import *`. This makes dependencies clear and avoids namespace conflicts.

- *Prefer method chaining for data operations:* Use pandas method chaining to create clear, readable pipelines. This keeps the logic transparent and mirrors the tidyverse approach in R.

- *Use one operation per line in chains:* When chaining operations, place each method on a new line. This makes debugging easier and helps others trace your logic step by step.

- *Include type hints where helpful:* For functions and complex operations, type hints improve code clarity and help catch errors early. Use them especially for file paths, DataFrames, and return values.

- *Structure scripts with clear sections:* Use comments with dashes or equals signs to create visual breaks between sections. This helps organize your analysis into logical chunks.

- *Use pathlib for file operations:* The `pathlib` module provides a modern, cross-platform way to handle file paths that works consistently across operating systems.

- *Keep code to 88 characters wide:* Line-wrapping helps with readability and makes version control differences much easier to follow. If a line is getting too long, split it logically across lines using Python's implicit line continuation within parentheses.

- *Use docstrings for functions:* Include clear docstrings that explain what the function does, its parameters, and return values. This is especially important when creating reusable functions.

- *Prefer f-strings for string formatting:* Use f-string syntax (`f"Population in {district}: {pop_count}"`) rather than older formatting methods. It's more readable and performant.

To see how these principles come together, unfold the example below. It shows a well-organized script that follows Python conventions: PEP 8 style, explicit imports, method chaining, clear section headers, and pathlib for file management. This structure improves readability, reduces bugs, and makes it easier to reuse or adapt the code in future workflows.

Unfold below to see these principles and style applied in practice, using DHIS2 data cleaning as an example.

```python
# Set up and data import -------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Define file paths using pathlib
data_dir = Path("data")
raw_file = data_dir / "dhis2_malaria_data.xlsx"

# Import raw DHIS2 data
raw_data = pd.read_excel(raw_file)

# Data cleaning and wrangling --------------------------------------------------

def clean_dhis2_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize DHIS2 malaria data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw DHIS2 data with French column names and period formatting
        
    Returns:
    --------
    pd.DataFrame
        Cleaned data with standardized columns and date components
    """
    
    # Month name mapping for French to numeric conversion
    month_mapping = {
        'Janvier': 1, 'Fevrier': 2, 'Mars': 3, 'Avril': 4,
        'Mai': 5, 'Juin': 6, 'Juillet': 7, 'Aout': 8,
        'Septembre': 9, 'Octobre': 10, 'Novembre': 11, 'Decembre': 12
    }
    
    return (df
        # Standardize column names
        .rename(columns={
            'OrganisationUnitLevel2': 'adm1',
            'OrganisationUnitLevel3': 'adm2', 
            'OrganisationUnitName': 'facility',
            'PeriodName': 'period'
        })
        # Split period into components
        .assign(
            month_name=lambda x: x['period'].str.split(' ').str[0],
            year=lambda x: (x['period']
                           .str.split(' ')
                           .str[1]
                           .astype(int))
        )
        # Convert month names to numbers
        .assign(
            month=lambda x: x['month_name'].map(month_mapping)
        )
        # Clean up intermediate columns
        .drop(columns=['period', 'month_name'])
        .dropna(subset=['month'])  # Remove rows with unmatched months
    )

# Apply cleaning function
processed_data = clean_dhis2_data(raw_data)

# Save processed data with meaningful filename
output_file = data_dir / "processed" / f"dhis2_cleaned_{pd.Timestamp.now().strftime('%Y%m%d')}.parquet"
processed_data.to_parquet(output_file, index=False)

# Generate basic summary for validation
print(f"Processed {len(processed_data):,} records")
print(f"Date range: {processed_data['year'].min()}-{processed_data['year'].max()}")
print(f"Unique facilities: {processed_data['facility'].nunique():,}")
```
