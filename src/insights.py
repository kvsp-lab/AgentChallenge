def generate_insights(df):
    insights = []
    
    if df.empty:
        return ["No transactions loaded to generate insights."]
        
    total_spend = df['Amount'].sum()
    cat_sums = df.groupby('Final Category')['Amount'].sum().sort_values(ascending=False)
    
    if not cat_sums.empty:
        top_cat = cat_sums.index[0]
        top_pct = (cat_sums.iloc[0] / total_spend) * 100
        insights.append(f"{top_cat} accounts for {top_pct:.0f}% of recorded spending.")
        
        if len(cat_sums) > 1:
            second_cat = cat_sums.index[1]
            insights.append(f"{second_cat} is your second-largest spending category.")
            
        needs_review_count = len(df[df['Review Status'] == 'Needs Review'])
        if needs_review_count > 0:
            insights.append(f"{needs_review_count} transactions require manual review.")
        else:
            insights.append("All transactions have been reviewed.")
            
        if 'Shopping' in cat_sums and 'Dining & Restaurants' in cat_sums:
            if cat_sums['Shopping'] > cat_sums['Dining & Restaurants']:
                insights.append("Shopping spending exceeds dining spending in this dataset.")
            else:
                insights.append("Dining spending exceeds shopping spending in this dataset.")
                
    return insights
