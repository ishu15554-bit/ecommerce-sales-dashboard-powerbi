# Import Required Libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import os

# Get Current Working Directory
current_path = os.getcwd()
file_path = os.path.join(current_path, "DATA", "eccomerce.csv")
print(file_path)
df = pd.read_csv(file_path)
# Preview Dataset
print(df.head())
# Check Dataset Shape
print(df.shape)
# Check Dataset Information
print(df.info())
# Generate Statistical Summary
print(df.describe())


# >>>>>Data Cleaning

# Check Missing Values
missing_value= df.isnull().sum()
print("Missing Values:")
print(missing_value)

# check duplicate value 
duplicates = df.duplicated().sum()
print("duplicates values")
print(duplicates)

# Change column names to uppercase
df.columns = df.columns.str.lower()

# >>>>> exploratory data analysis(EDA)

# Q1 WHICH PRODUCT CATEGORY GENRATE THE HIGHEST TOTAL REVENUE?

highest_revenue = df.groupby( "product_category")['revenue'].sum().idxmax()
print(highest_revenue)
print()

# Q2. Which region generated the highest total revenue?
region_highest = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
print(region_highest)
print()

#  Q3. Which payment method is used the most?
payment_method_use = df.groupby("payment_method")["payment_method"].value_counts().sort_values(ascending=False)
print(payment_method_use)
print()

#  Q4. Which product category has the highest number of orders?
highest_order = df.groupby('product_category')['order_id'].count().idxmax()
print(highest_order)
print()

# Q5.  Product Categories Ranked by Total Revenue
top_revenue = df.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
print(top_revenue)
print()

# Q6. Which Payment Method Generated the Highest Revenue?
payment_revenue = df.groupby("payment_method")["revenue"].sum().sort_values(ascending=False)
print(payment_revenue)

# Visualization
plt.figure(figsize=(8,5))
payment_revenue.plot(kind="bar")
plt.title("Revenue by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Total Revenue")
plt.savefig("IMAGES/q6_payment_method_revenue.png", dpi=300)
plt.show()

# Q7. Monthly Revenue Trend

# Convert order_date to datetime
df["order_date"] = pd.to_datetime(df["order_date"])

# Extract Month Number and Month Name
df["Month_No"] = df["order_date"].dt.month
df["Month"] = df["order_date"].dt.month_name()

# Calculate Monthly Revenue
monthly_revenue = ( df.groupby(["Month_No", "Month"])["revenue"].sum().reset_index().sort_values("Month_No"))
print(monthly_revenue)

# Visualization
plt.figure(figsize=(10,5))
plt.plot(
    monthly_revenue["Month"],
    monthly_revenue["revenue"],
    marker="o",
    linewidth=2
)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.tight_layout()
plt.savefig("IMAGES/q7_monthly_revenue.png", dpi=300)
plt.show()


# ✅ Q8. Revenue distribution by payment method.
payment_rev_distribution = df.groupby("payment_method")['revenue'].sum()
print(payment_rev_distribution)
print()

#visualisation
plt.figure(figsize=(8,8))
payment_rev_distribution.plot(kind = "pie",autopct="%1.1f%%",
    startangle=90)
plt.title("Revenue Distribution by Payment Method")
plt.ylabel("")
plt.tight_layout()
plt.savefig("IMAGES/q8_payment_method_pie.png", dpi=300)
plt.show()

#  Q9. Which region has the highest average order value? 
region_average_order = df.groupby('region')['revenue'].mean()
print(region_average_order)

#visualisation
plt.figure(figsize=(10,8))
region_average_order.plot(kind="bar")
plt.title("average order value by region")
plt.xlabel("region")
plt.ylabel("average revenue")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("IMAGES/q9_average_order_region.png", dpi=300)
plt.show()

#  Q10. Which Product Category Has the Highest Average Customer Rating?
average_customer_rating = ( df.groupby("product_category")["customer_rating"].mean())
print(average_customer_rating)
print()

highest_category = average_customer_rating.idxmax()
print("Highest Average Rating Category :", highest_category)

# visiualistion
plt.figure(figsize=(9, 6))
average_customer_rating.plot(kind="bar")
plt.title("Average Customer Rating by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Customer Rating")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("IMAGES/q10_average_customer_rating.png", dpi=300)
plt.show()

# ✅ Q11. Top 10 customers by revenue.
top_customer_revenue = df.groupby("customer_id")['revenue'].sum().sort_values(ascending=False).head(10)
print(top_customer_revenue)
print()

plt.figure(figsize=(10,6))

top_customer_revenue.plot(kind="barh")

plt.title("Top 10 Customers by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Customer ID")
plt.grid(axis="x")
plt.tight_layout()

plt.savefig("IMAGES/q11_top10_customers_revenue.png", dpi=300)

plt.show()

#Q12. Which Region Generates the Highest Revenue Per Order?
total_revenue = df.groupby("region")['revenue'].sum()
print(total_revenue)
print()
total_order = df.groupby("region")["order_id"].count()
print(total_order)
print()
revenue_per_order = total_revenue/ total_order
print(revenue_per_order.idxmax())

plt.figure(figsize=(8,6))
revenue_per_order.plot(kind="bar")
plt.title("Average Revenue Per Order by Region")
plt.xlabel("Region")
plt.ylabel("Revenue Per Order")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("IMAGES/q12_revenue_per_order_region.png", dpi=300)
plt.show()


# Q13. Which Product Category Has the Longest Average Delivery Time?

average_delivery_time = (df.groupby("product_category")["delivery_days"].mean().sort_values(ascending=False))
print(average_delivery_time)
print("Category with Longest Delivery Time:")
print(average_delivery_time.idxmax())

# visualisation
plt.figure(figsize=(8,6))
average_delivery_time.plot(kind="bar")
plt.title("Average Delivery Time by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Delivery Days")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig("IMAGES/q13_average_delivery_time.png", dpi=300)

plt.show()

# Q14. Correlation Between Numerical Features
plt.figure(figsize=(8,6))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap( correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Between Numerical Features")

plt.tight_layout()

plt.savefig("IMAGES/q14_correlation_heatmap.png", dpi=300)

plt.show()

# Q15. Distribution of Customer Ratings

plt.figure(figsize=(8,6))
plt.hist( df["customer_rating"], bins=10, edgecolor="black")

plt.title("Distribution of Customer Ratings")
plt.xlabel("Customer Rating")
plt.ylabel("Number of Customers")

plt.grid(axis="y")
plt.tight_layout()
plt.savefig("IMAGES/q15_customer_rating_distribution.png", dpi=300)

plt.show()


# >>>>>>BONUS FEATURE : SMART OFFER PLANNER
# ==========================================================

print("           SMART OFFER PLANNER")

# Take product category as input
category = input("Enter Product Category : ").strip()

# Check if category exists
if category not in df["product_category"].unique():
    print("\nCategory not found!")
else:

    # Filter selected category
    category_data = df[df["product_category"] == category]

    # Create month column if not available
    if "Month" not in category_data.columns:
        category_data["Month"] = pd.to_datetime(
            category_data["order_date"]
        ).dt.month_name()

    # Monthly summary
    monthly_summary = (
        category_data.groupby("Month")
        .agg(
            Average_Revenue=("revenue", "mean"),
            Average_Discount=("discount", "mean"),
            Average_Rating=("customer_rating", "mean"),
            Average_Delivery=("delivery_days", "mean")
        )
        .round(2)
    )

    # Best month
    best_month = monthly_summary["Average_Revenue"].idxmax()

    print("\nBest Performing Month :", best_month)

    print("\nBusiness Summary")
    print(monthly_summary.loc[best_month])

    print("\nSuggested Business Strategy")

    discount = monthly_summary.loc[best_month, "Average_Discount"]

    if discount <= 10:
        print("- Low discount generated good revenue.")
        print("- Consider keeping offers around 5% to 10%.")

    elif discount <= 20:
        print("- Moderate discount performed well.")
        print("- Consider offers between 10% and 20%.")

    else:
        print("- High discounts were used.")
        print("- Review profit before giving large discounts.")

    rating = monthly_summary.loc[best_month, "Average_Rating"]

    if rating >= 4.5:
        print("- Customer satisfaction is excellent.")

    elif rating >= 4:
        print("- Customer satisfaction is good.")

    else:
        print("- Customer satisfaction needs improvement.")

    delivery = monthly_summary.loc[best_month, "Average_Delivery"]

    if delivery <= 3:
        print("- Delivery performance is fast.")

    else:
        print("- Delivery process can be improved.")

print("="*55)