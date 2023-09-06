# import the libraries
import pandas as pd

pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

#########################
# TASK 1: Prepare the Data
#########################

# Step 1: read the armut_data.csv file
df_ = pd.read_csv("elfakbaba/pythonproject/armut_data.csv")
df = df_.copy()
df.head()

# Step 2: The ServicID represents a different service for each specific CategoryID.
# Create a new variable to represent the services by combining serviceId and CategoryID with "_".
#######################################################################################################

# NOTE:
# Service is the service received under the "Category".
# For example;
# Under the "Cleaning" category, "Seat Cleaning" is a service.
# Under the category of "repair", "faucet repair" is a service


df.values

# for row in df.values = go through all the line values.
# the values of each row are;
# row[0] = userid
# row[1] = serviceid
# row[2] = categoryid
# row[3] = createdate

df.values

df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

# --------------------------------------------------------------------------------------------------------------------------
# Step 3: The data set consists of the date and time of receipt of the services, any basket definition (invoice, etc. ) are not available.
# A basket (invoice, etc.) to be able to apply Association Rule Learning.) the definition needs to be created.

# The basket definition here is the services that each customer receives monthly.
# For example, 25446 customers with ID 8 of 2017.a basket of 4_5, 48_5, 6_7, 47_7 services received in the month; 9th of 2017.in the month
# the 17_5, 14_7 services it receives refer to another basket.

# Baskets must be identified with a unique ID. To do this, first create a new date variable that contains only the year and month.
# Combine the UserID and the date variable you just created with the user-specific "_" and assign it to a new variable named ID.
############################################################################################################################################

# first of all, we need to convert the createdate variable to the datatime type.
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()

# We don't have a basket or a bill, that's why;
# We are trying to create a basket.
# Let's create this basket for a month.
# So, let's divide the date variable into month and year.

df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df["NEW_DATE"].head()

# Combine the UserID and the date variable you just created with the user-specific "_" and assign it to a new variable named ID.

# then, by combining row[0] userid and row[5] new_date, we actually create a basket.
# those belonging to the same sepetid are services purchased in one basket, that is, in 1 month.
df.head()
df.values
df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

df.sort_values("SepetID").head()  # aynı müşterinin aldıgı hizmetler farklı satırlarda olmak üzere görülür.

###############################################
# TASK 2: Create Rules of Association
###############################################

############################################################################################################
# Step 1:Create the basket service pivot table as follows.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..
############################################################################################################################################

# in fact, what is required of us at this step is to create an item_set. so;
# let there be baskets in the rows,
# let the column variables be of service,
# let the information be expressed whether the service values are taken at their intersections.

# "sepetid", "hizmet"  we do gropby
# we calculate the "number of services" (count).
# we are pivoting with unstack, so we are bringing it exactly to the format we want.
# unstack -> places baskets in the row, services in the column, the number of services, which is the third breakdown, at their intersections, that is, in cells.
# we fill in the nan values with 0 (fillna(0))
# finally, the number of services may be different from 1.
# we are not interested in the number, all we are interested in here is whether he received the service.
# "applymap(lambda x: 1 if x > 0 else 0" with applymap, go through all the cells and write 1 if it is greater than 0, if not 0
# in this way, we fill in the cells with information that has not been received.

df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().head()
df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().head()
df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0)
invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0)
invoice_product_df.head()

########################################################################
# Step 2: Create the rules of association.
########################################################################

# The a priori algorithm prepares the table in order to create an association rule.
# min_support ister.

frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
frequent_itemsets.tail()

# frequent_itemsets = fpgrowth(invoice_product_df, min_support=0.01, use_colnames=True)


# We must establish the rules of association.

# The rules calculate the lift,confidence, support values.
# antecedents = X
# consequents = Y
# We calculate the rules a priori based on the table we have prepared.

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

# thus, we create the rules of association.

################################################################################################################################################
# Adım 3: using the arl_recommender function, make a service recommendation to a user who has received the 2_0 service within the last 1 month.
################################################################################################################################################

# Let's create a recommendation:

# rec_count: how many product recommendations do you want?
# product_id: which product do you want advice for?
# rules_df: a list of rules that I will use.


rules.sort_values("lift", ascending=False).head()


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # it sorts the rules from large to small. (in order to catch the most compatible first product)
    # according to confidence, it also depends on the sortable initiative.
    recommendation_list = []  # tavsiye edilecek ürünler için bos bir liste olusturuyoruz.
    # antecedents: X
    # because it is called items, it brings it as a frozenset. combines index and service.
    # i: index
    # product: X, i.e. the service that asks for suggestions
    for i, product in enumerate(sorted_rules["antecedents"]):  # enumerate
        for j in list(product):  # sightseeing in services(product):
            if j == product_id:  # if the recommended product is caught:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
                # you were holding the index information with I. Add the consequences(Y) value in this index information to the October list.

    # to prevent recurrence in the list of recommendations:
    # for example, in combinations of 2 and 3, the same product may have fallen back on the list, such as;
    # we take advantage of the unique feature of the dictionary structure.
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]  # :rec_count Bring recommended products up to the requested number.


# let's have 1 recommendation for the 2_0 service:
arl_recommender(rules, "2_0", 1)

sorted_rules = rules.sort_values("lift", ascending=False).head()
sorted_rules["antecedents"].items()
