# Feature Statistics Report (PA)

Generated from processed CSV files in `data/processed/`.

---

## Table D4 — User Auxiliary Information After Min-Max Normalization (PA)

- Source: `pa_users_feat.csv`
- Features: 33 columns
- Users: 320,212
- Normalization: Min-Max → all values in [0, 1]
- Statistics: Mean, Standard Deviation, 25th/50th/75th Percentile

![Table D4](table_d4_user_features.png)

### Feature list:
- compliment_cool
- compliment_cute
- compliment_funny
- compliment_hot
- compliment_list
- compliment_more
- compliment_note
- compliment_photos
- compliment_plain
- compliment_profile
- compliment_writer
- votes_cool
- votes_funny
- votes_useful
- fans
- yelping_since_year
- yelping_since_month
- yelping_since_day
- elite_2005
- elite_2006
- elite_2007
- elite_2008
- elite_2009
- elite_2010
- elite_2011
- elite_2012
- elite_2013
- elite_2014
- elite_2015
- elite_2016
- elite_2017
- elite_2018
- elite_None

---

## Table D5 — Business Auxiliary Information (Attributes)

- Source: `pa_biz_features.csv` (attr_* columns)
- Attributes: 87 columns (77 with data, 10 NaN)
- Businesses: 31,663
- Encoding: Binary one-hot (0/1)
- Statistics: PA Sum (count of 1s), PA Mean (proportion)

![Table D5](table_d5_biz_attributes.png)

---

## Table D6 — Business Auxiliary Information (Categories)

- Source: `pa_biz_features.csv` (cat_* columns)
- Categories: 1149 columns (1149 with data, 0 NaN)
- Businesses: 31,663
- Encoding: Multi-label one-hot (0/1)
- Statistics: PA Sum (count of 1s), PA Mean (proportion)

![Table D6](table_d6_biz_categories.png)
