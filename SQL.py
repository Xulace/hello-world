
# coding: utf-8

# In[88]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import seaborn as sns
import pandas as pd
import sqlalchemy
sns.set()
matplotlib.rcParams['figure.dpi'] = 144


# In[2]:


from static_grader import grader


# # SQL Miniproject
# 

# ## Introduction
# 

# The city of New York does restaurant inspections and assigns a grade. Inspections data for the last 4 years are available on s3 as an SQLite database, which you can import in the next few cells. These were extracted from a set of csv files and an XLS file, as described in the <b>How we loaded the data</b> section
# 
# 
# The raw data can be found [here](https://s3.amazonaws.com/dataincubator-course/coursedata/nyc_inspection_data.zip) and can be useful to look at. The file `RI_Webextract_BigApps_Latest.xls` contains a description of each of the data files and what the columns mean.

# In[3]:


get_ipython().system("aws s3 sync s3://dataincubator-course/coursedata/ . --exclude '*' --include 'nyc_inspection.db'")


# In[4]:


#This will load the pre-existing tables
get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite:///nyc_inspection.db')


# To see what tables are in the database:

# In[5]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM sqlite_master WHERE "type"=\'table\';')


# And to look at the format of an individual table (note that you may need to change types to get the answers in the right form):

# In[6]:


get_ipython().run_cell_magic('sql', '', 'PRAGMA table_info(webextract)')


# ## SQLite3
# 

# The project should be written in SQL. Between SQLite and PostgreSQL we recommend SQLite for this project.  You can use the SQLite command prompt by running this command in bash
# ```bash
# sqlite3 cmd "DROP TABLE IF EXISTS writer;\
# CREATE TABLE IF NOT EXISTS writer (first_name, last_name, year);\
# INSERT INTO writer VALUES ('William', 'Shakespeare', 1616);\
# INSERT INTO writer VALUES ('Francis', 'Fitzgerald', 1896);\
# \
# SELECT * FROM writer;\
# "
# ```
# Alternatively, you can run bash commands in a Jupyter notebook by prepending the `!` in a code cell (notice that we conveniently get the output displayed

# In[7]:


get_ipython().system('sqlite3 cmd """DROP TABLE IF EXISTS writer;CREATE TABLE IF NOT EXISTS writer (first_name, last_name, year);INSERT INTO writer VALUES (\'William\', \'Shakespeare\', 1616);INSERT INTO writer VALUES (\'Francis\', \'Fitzgerald\', 1896);SELECT * FROM writer;"""')


# Finally, we use the [`ipython-sql` extension](https://github.com/catherinedevlin/ipython-sql#ipython-sql) by first loading this extension and then running our code with the "magic" command in the first line
# ```python
# %%sql sqlite://
# ```
# Notice that the output table is formatted nicely as a nice HTML table.
# 
# This is our recommended technique.  However, the grader is expecting python objects and you may need to use list comprehensions to reformat this output

# In[98]:


get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS writer;\nCREATE TABLE IF NOT EXISTS writer (first_name, last_name, year);\nINSERT INTO writer VALUES ('William', 'Shakespeare', 1616);\nINSERT INTO writer VALUES ('Francis', 'Fitzgerald', 1896);\n\nSELECT * FROM writer;")


# In[47]:


result = _
#This captures the output of the previous cell


# In[87]:


type(result)


# In[96]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///nyc_inspection.db')
df = pd.read_sql_table('writer', engine)
df


# ## How we loaded the data
# 

# For future reference, here is how you can load data in to SQL (with examples).  If you have a csv file you created with something like
# 
# ```
# !printf "Name,Age\nAlice,3\nBob,10" > sample.csv.nogit
# ```
# 
# 
# Then SQLite has a convenient [`.import` function](https://sqlite.org/cli.html#csv_import) which can create tables from `.csv` files.
# 
# ```bash
# sqlite> .import sample.csv.nogit sample
# sqlite> SELECT * FROM sample;
# ```
# 
# The files may contain badly formatted text.  Unfortunately, this is all too common.  As a stop gap, remember that [`iconv`](https://linux.die.net/man/1/iconv) is a Unix utility that can convert files between different text encodings.
# 
# Alternatively, you can also read csv files using pandas and convert that into SQL via some SQL magic (this is what we actually did).
# 
# ```
# import pandas as pd
# sample = pd.read_csv('sample.csv.nogit')
# %sql DROP TABLE IF EXISTS sample
# %sql PERSIST sample
# %sql SELECT * FROM sample;
# ```

# ## Question 1: Null entries
# 

# Return the number of inspections (`CAMIS`, `INSPDATE` pairs) that do not have a score - i.e. where none of the rows with those (`CAMIS`, `INSPDATE`) values has a score. Remove the corresponding rows from the data set for the rest of the questions in the assignment.
# 
# **Question:** How else might we have handled this?

# In[131]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS null_results;\nCREATE TABLE IF NOT EXISTS null_results AS\n    SELECT CAMIS, INSPDATE\n    FROM webextract\n    WHERE (SCORE IS NULL);')


# In[126]:


#from sqlalchemy import create_engine
#engine = create_engine('sqlite:///nyc_inspection.db')
null_results = pd.read_sql_table('null_results', engine)
null_results


# In[128]:


len(null_results)


# In[129]:


def null_entries():
    return len(null_results)

grader.score('sql__null_entries', null_entries)


# In[136]:


get_ipython().run_cell_magic('sql', '', 'DELETE FROM webextract WHERE SCORE IS NULL;')


# ## Question 2: Score by zip code
# 

# Return a list of tuples of the form:
# 
#     (zip code, mean score, number of restaurants)
# 
# for each of the 87 zip codes in the city with over 100 restaurants. Use the score from the latest inspection date for each restaurant. Sort the list in ascending order by mean score.
# 
# **Note:** There is an interesting discussion here about what the mean score *means* in this data set. Think about what we're actually calculating - does it represent what we're trying to understand about these zip codes?
# 
# What if we use the average of a restaurant's inspections instead of the latest?
# 
# **Checkpoints:**
# - Total unique restaurants: 24,361;
# - Total restaurants in valid zip codes: 19,172
# 

# In[372]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS zip_scores;\nCREATE TABLE IF NOT EXISTS zip_scores AS\n    SELECT ZIPCODE, SCORE, MAX(GRADEDATE) AS recent_date\n    FROM webextract\n    GROUP BY CAMIS;')


# In[373]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS zipcode;\nCREATE TABLE IF NOT EXISTS zipcode AS\n    SELECT ZIPCODE, AVG(SCORE), COUNT(*)\n        FROM zip_scores\n        GROUP BY ZIPCODE\n        HAVING COUNT(*) > 100')


# In[374]:


zipcode = pd.read_sql_table('zipcode', engine)
zipcode['ZIPCODE'] = zipcode['ZIPCODE'].astype(int)
#zipcode['ZIPCODE'] = zipcode['ZIPCODE'].astype(str)
zipcode


# In[365]:


zipcode.sort_values(by=['AVG(SCORE)'])


# In[366]:


def score_by_zipcode():
    return zipcode.sort_values(by=['AVG(SCORE)'])
#[("11201", 9.81739130434783, 345)] * 87

grader.score('sql__score_by_zipcode', score_by_zipcode)


# ## Question 3: Mapping scores
# 

# The above are not terribly enlightening.  Use [CartoDB](http://cartodb.com/) to produce a map of average scores by zip code.  You can sign up for a free trial.
# 
# You will have to use their wizard to plot the data by [zip code](https://carto.com/learn/guides/analysis/georeference). You will need to specify "USA" in the country field.  Then use the "share" button to return a link of the form [https://x.cartodb.com/](https://x.cartodb.com/).
# 
# **For fun:** How do JFK, Brighton Beach, Liberty Island (home of the Statue of Liberty), Financial District, Chinatown, and Coney Island fare?
# 
# **For more fun:** Plot restaurants as pins on the map, allowing the user to filter by "low", "middling", or "high"-scoring restaurants. You can use a CASE WHEN statement to create the different groups based on score thresholds.

# In[ ]:


def score_by_map():
    # must be url of the form https://x.cartodb.com/...
    return "https://cartodb.com"

grader.score('sql__score_by_map', score_by_map)


# ## Question 4: Score by borough

# Return a list of tuples of the form:
# 
#     (borough, mean score, number of restaurants)
# 
# for each of the city's five boroughs. Use the latest score for each restaurant. Sort the list in ascending order by grade.
# 
# **Hint:** You will have to perform a join with the `boroughs` table. The borough names should be reported in ALL CAPS.
# 
# **Checkpoint:**
# - Total restaurants in valid boroughs: 24,350

# In[156]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS boro_scores;\nCREATE TABLE IF NOT EXISTS boro_scores AS\n    SELECT BORO, SCORE, MAX(GRADEDATE) AS recent_date\n    FROM webextract\n    GROUP BY CAMIS;')


# In[170]:


get_ipython().run_cell_magic('sql', '', 'SELECT boroughs.name, boro_scores.SCORE\nFROM boro_scores\nLEFT JOIN boroughs ON boro_scores.BORO=boroughs.id;')


# In[171]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS boros;\nCREATE TABLE IF NOT EXISTS boros AS\n    SELECT boroughs.name, boro_scores.SCORE\n    FROM boro_scores\n    LEFT JOIN boroughs ON boro_scores.BORO=boroughs.id;')


# In[172]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS boro_avg;\nCREATE TABLE IF NOT EXISTS boro_avg AS\n    SELECT name, AVG(SCORE), COUNT(*)\n        FROM boros\n        GROUP BY name')


# In[189]:


boro = pd.read_sql_table('boro_avg', engine)
boro = boro.drop(boro.index[0])
boro = boro.sort_values(by=['AVG(SCORE)'])
boro


# In[190]:


def score_by_borough():
    return boro
#[("MANHATTAN", 10.7269875502402, 10201)] * 5

grader.score('sql__score_by_borough', score_by_borough)


# ## Question 5: Violations by cuisine
# 

# We want to look at violations themselves now.  We'll need to think more carefully about what we're measuring, since most restaurants have many inspections with possibly multiple violations per inspection, or long stretches of inspections with no violations.
# 
# There are many ways to deal with this normalization issue, but we'll go with a fairly straightforward one: dividing the number of violations by the length of time (in years) the restaurant has been open.  As a proxy for the length, we'll look at the difference between the oldest and newest inspection date, treating anything less than 30 days as 30 days (to account for those that were only inspected once, we'll assume everything was open for at least a month).
# 
# Since there are so many restaurants, we'll group them by cuisine and do a weighted average by computing 
# 
#     (total violations for a cuisine) / (total restaurant-years for that cuisine)
# 
# Return a list of 75 tuples of the form
# 
#     (cuisine name, reports per restaurant-year)
#     
# for cuisines with at least 100 violations total, ordered by increasing number of reports per restaurant-year
#     
# **Note:** This isn't the only way to normalize things.  How would other ways affect the computation?  If you similarly wanted to compute an average score by cuisine, how might you go about doing that?
#     
# **Checkpoint:**
# - Total entries from valid cuisines: 522,410

# In[198]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS tot_cuisines;\nCREATE TABLE IF NOT EXISTS tot_cuisines AS\n    SELECT CAMIS, CUISINECODE, GRADEDATE\n    FROM webextract')


# In[199]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS tot_cuisines2;\nCREATE TABLE IF NOT EXISTS tot_cuisines2 AS\n    SELECT tot_cuisines.CAMIS, tot_cuisines.GRADEDATE, cuisine.CODEDESC\n    FROM tot_cuisines\n    LEFT JOIN cuisine ON tot_cuisines.CUISINECODE=cuisine.CUISINECODE;')


# In[224]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS tot_cuisines3;\nCREATE TABLE IF NOT EXISTS tot_cuisines3 AS\n    SELECT CAMIS, CODEDESC, COUNT(*), JULIANDAY(MAX(GRADEDATE)) - JULIANDAY(MIN(GRADEDATE)) AS DIFFERENCE FROM tot_cuisines2\n    GROUP BY CAMIS;')


# In[229]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS camis_avg;\nCREATE TABLE IF NOT EXISTS camis_avg AS\n    SELECT CODEDESC, COUNT(*) - 1 AS AVG FROM tot_cuisines3\n    GROUP BY CAMIS;')


# In[225]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM tot_cuisines3\nLIMIT 3;')


# In[247]:


camis_avg = pd.read_sql_table('tot_cuisines3', engine)
camis_avg


# In[251]:


cuisine_sum = camis_avg.groupby('CODEDESC').sum()
cuisine_sum


# In[255]:


cuisine_sum['AVG'] = cuisine_avg['COUNT(*)']/(cuisine_avg['DIFFERENCE']/365)
cuisine_sum


# In[259]:


cuisine_avg = cuisine_sum.drop(columns=['CAMIS', 'DIFFERENCE'])
cuisine_avg


# In[287]:


cuisavg_100 = cuisine_avg.loc[cuisine_avg['COUNT(*)'] > 100]
cuisavg_100


# In[288]:


cuisavg_100 = cuisavg_100.drop(columns=['COUNT(*)'])


# In[289]:


cuisavg_100 = cuisavg_100.sort_values(by=['AVG'])
cuisavg_100


# In[290]:


cuisavg_100['AVG'] = cuisavg_100['AVG'].astype(str)
cuisavg_100


# In[291]:


def score_by_cuisine():
    return cuisavg_100
#[("French", 20.3550686378036)] * 75

grader.score('sql__score_by_cuisine', score_by_cuisine)


# ## Question 6: Specific violations by cuisine

# Which cuisines tend to have a disproportionate number of what which violations? Answering this question isn't easy because you have to think carefully about normalizations.
# 
# 1. More popular cuisine categories will tend to have more violations just because they represent more restaurants.
# 2. Similarly, some violations are more common.  For example, knowing that "Equipment not easily movable or sealed to floor" is a common violation for Chinese restaurants is not particularly helpful when it is a common violation for all restaurants.
# 
# The right quantity is to look at is the conditional probability of a specific type of violation given a specific cuisine type and divide it by the unconditional probability of the violation for the entire population. Taking this ratio gives the right answer.  Return the 20 highest ratios of the form:
# 
#     ((cuisine, violation), ratio, count)
# 
# **Hint:**
# 1. You might want to check out this [Stack Overflow post](http://stackoverflow.com/questions/972877/calculate-frequency-using-sql).
# 2. The definition of a violation changes with time.  For example, 10A can mean two different things "Toilet facility not maintained ..." or "Vermin or other live animal present ..." when things were prior to 2003. To deal with this, you should limit your analysis to violation codes with end date after Jan 1, 2014. (This end date refers to the validity time ranges in `Violation.txt`).
# 3. The ratios don't mean much when the number of violations of a given type and for a specific category are not large (why not?).  Be sure to filter these out.  We chose 100 as our cutoff.
# 
# **Checkpoint:**
# - Top 20 ratios mean: 2.360652529900757

# In[315]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS viol_cuisines;\nCREATE TABLE IF NOT EXISTS viol_cuisines AS\n    SELECT CAMIS, CUISINECODE, VIOLCODE, GRADEDATE\n    FROM webextract')


# In[316]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM viol_cuisines\nLIMIT 5;')


# In[333]:


get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS violcode;\nCREATE TABLE IF NOT EXISTS violcode AS\n    SELECT *\n    FROM violations\n    GROUP BY VIOLATIONCODE\n    HAVING ENDDATE > '2014-01-01'")


# In[334]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM violcode\nLIMIT 5;')


# In[338]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS violcode2;\nCREATE TABLE IF NOT EXISTS violcode2 AS\n    SELECT * FROM violcode')


# In[340]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM violcode2\nLIMIT 5;')


# In[330]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS viol_cuisines2;\nCREATE TABLE IF NOT EXISTS viol_cuisines2 AS\n    SELECT viol_cuisines.CAMIS, cuisine.CODEDESC, viol_cuisines.VIOLCODE\n    FROM viol_cuisines\n    LEFT JOIN cuisine ON viol_cuisines.CUISINECODE=cuisine.CUISINECODE;')


# In[329]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM viol_cuisines2\nLIMIT 5;')


# In[343]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS viol_cuisines3;\nCREATE TABLE IF NOT EXISTS viol_cuisines3 AS\n    SELECT viol_cuisines2.CAMIS, viol_cuisines2.CODEDESC, viol_cuisines2.VIOLCODE, violcode2.VIOLATIONDESC\n    FROM viol_cuisines2\n    LEFT JOIN violcode2 ON viol_cuisines2.VIOLCODE=violcode2.VIOLATIONCODE;')


# In[344]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM viol_cuisines3\nLIMIT 5;')


# In[395]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS viol_cuisines4;\nCREATE TABLE IF NOT EXISTS viol_cuisines4 AS\n    SELECT CODEDESC, VIOLCODE, VIOLATIONDESC, COUNT(*)\n        FROM viol_cuisines3\n        GROUP BY CODEDESC, VIOLCODE, VIOLATIONDESC\n        HAVING COUNT(VIOLCODE) > 100;')


# In[414]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS viol_freq;\nCREATE TABLE IF NOT EXISTS viol_freq AS\n    SELECT  A.CODEDESC, A.VIOLCODE, A.COUNT1 * 1.0 / B.COUNT2 AS FREQ\n    FROM    (\n            SELECT CODEDESC, VIOLCODE, COUNT(*) As COUNT1\n            FROM viol_cuisines3\n            GROUP BY CODEDESC, VIOLCODE\n            ) AS A\n            INNER JOIN (\n                SELECT CODEDESC, COUNT(*) As COUNT2\n                FROM viol_cuisines3\n                GROUP BY CODEDESC\n                ) As B\n                On A.CODEDESC = B.CODEDESC')


# In[415]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM viol_freq\nLIMIT 5;')


# In[409]:


get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS viol_freq2;\nCREATE TABLE IF NOT EXISTS viol_freq2 AS\n    SELECT viol_freq.CODEDESC, violcode2.VIOLATIONDESC, viol_freq.FREQ\n    FROM viol_freq\n    LEFT JOIN violcode2 ON viol_freq.VIOLCODE=violcode2.VIOLATIONCODE;')


# In[410]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM viol_freq2\nLIMIT 5;')


# In[ ]:


def violation_by_cuisine():
    return [(("Café/Coffee/Tea",
              "Toilet facility not maintained and provided with toilet paper; "
              "waste receptacle and self-closing door."),
             1.87684775827172, 315)] * 20

grader.score('sql__violation_by_cuisine', violation_by_cuisine)


# *Copyright &copy; 2016 The Data Incubator.  All rights reserved.*
