def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    category_sums = {}
    category_counts = {}

    for cat, target in zip(categories, targets):
        category_sums[cat] = category_sums.get(cat, 0.0) + target
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
    category_means = {
        cat: category_sums[cat] / category_counts[cat] 
        for cat in category_sums
    }
    
    return [category_means[cat] for cat in categories]
