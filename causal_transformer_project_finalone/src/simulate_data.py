import numpy as np
import pandas as pd

def simulate_data(n_individuals=10000, n_timepoints=10, seed=42):
    """ Data simulation code: Emulating Acute illness and Nutrition status 
    across different regions """
    np.random.seed(seed)

    # Constants
    sites = [
        "Kampala", "Banfora", "Dhaka", "Karachi", "Blantyre",
        "Matlab", "Nairobi", "Kilifi", "Migori"
    ]
    country_map = {
        "Kilifi": "Kenya", "Nairobi": "Kenya", "Migori": "Kenya",
        "Dhaka": "Bangladesh", "Matlab": "Bangladesh",
        "Karachi": "Pakistan",
        "Kampala": "Uganda",
        "Banfora": "Burkina Faso",
        "Blantyre": "Malawi"
    }
    region_map = {
        "Kenya": "Sub-Saharan Africa", "Uganda": "Sub-Saharan Africa", "Malawi": "Sub-Saharan Africa",
        "Bangladesh": "South East Asia", "Pakistan": "South East Asia",
        "Burkina Faso": "Sub-Saharan Africa"
    }
    age_bins = [0, 6, 12, 18, 24, 30, 36.1]
    age_labels = [
        "0-6 months", "6-12 months", "12-18 months", "18-24 months",
        "24-30 months", "30-36 months"
    ]

    # Generate baseline data
    record_ids = np.arange(1, n_individuals + 1)
    sexes = np.random.choice(["Male", "Female"], size=n_individuals)
    sites_sampled = np.random.choice(sites, size=n_individuals)
    agemons_baseline = np.random.uniform(2, 23, size=n_individuals)

    data = []

    for i in range(n_individuals):
        # Determine dropout (right censoring)
        dropout_time = np.random.randint(6, n_timepoints + 1)  # 6 to 10
        censored_flag = int(dropout_time == n_timepoints)
        timepoint_censored = np.nan if censored_flag else dropout_time

        for t in range(n_timepoints):
            missed_visit = int(t >= dropout_time)
            agemons = agemons_baseline[i] + t
            age_group = pd.cut([agemons], bins=age_bins, labels=age_labels, right=False)[0]
            sex = sexes[i]
            site = sites_sampled[i]
            country = country_map[site]
            region = region_map[country]

            # Time-varying covariates
            nutri_counsel = np.random.choice(["Yes", "No"], p=[0.7, 0.3])
            poor_feeding = np.random.choice(["Yes", "No"], p=[0.2, 0.8])
            acutely_ill = np.random.choice(["Yes", "No"], p=[0.3, 0.7])
            onoutpfeed_prog = np.random.choice(["No Feeding program", "RUTF for SAM", "RUSF for MAM"])

            # Compute pscore
            logit_pscore = (
                -0.5 +
                0.05 * agemons +
                0.4 * (sex == "Female") +
                0.3 * (region == "Sub-Saharan Africa") +
                0.6 * (nutri_counsel == "Yes") +
                0.7 * (poor_feeding == "Yes") +
                0.8 * (acutely_ill == "Yes")
            )
            pscore = 1 / (1 + np.exp(-logit_pscore))

            # Sample treatment
            binary_treatment = np.random.binomial(1, pscore)

            # Compute IPTW
            iptw = 1 / pscore if binary_treatment == 1 else 1 / (1 - pscore)

            # Simulated ITE for weight gain
            ite_weight = (
                0.1 * t
                + 0.5 * (sex == "Female")
                + 0.3 * (region == "Sub-Saharan Africa")
                + 0.4 * (nutri_counsel == "Yes")
                + 0.6 * (poor_feeding == "Yes")
                + 0.7 * (acutely_ill == "Yes")
                + np.random.normal(0, 0.5)
            )

            ite_feed = (
                -1 + 0.15 * t
                + 0.2 * (region == "Sub-Saharan Africa")
                + 0.4 * (nutri_counsel == "Yes")
                + 0.8 * (poor_feeding == "Yes")
                + 1.0 * (acutely_ill == "Yes")
                + np.random.normal(0, 0.3)
            )

            # Continuous outcome
            base_weight_gain = np.random.normal(loc=10, scale=8)
            pct_weight_gain_0 = base_weight_gain
            pct_weight_gain_1 = base_weight_gain + ite_weight
            pct_weight_gain_factual = pct_weight_gain_1 if binary_treatment else pct_weight_gain_0
            pct_weight_gain_counterfactual = pct_weight_gain_0 if binary_treatment else pct_weight_gain_1

            # Categorical outcome
            def feed_outcome_logit(val):
                if val < -3:
                    return "SAM"
                elif -3 <= val < -2:
                    return "MAM"
                else:
                    return "Normal"

            feed_outcome_0 = feed_outcome_logit(np.random.normal(-1 + ite_feed / 2, 1))
            feed_outcome_1 = feed_outcome_logit(np.random.normal(-1 + ite_feed / 2 + 0.5, 1))
            feed_outcome_factual = feed_outcome_1 if binary_treatment else feed_outcome_0
            feed_outcome_counterfactual = feed_outcome_0 if binary_treatment else feed_outcome_1

            data.append({
                "record_id": record_ids[i],
                "timepoint": t,
                "agemons": round(agemons, 2),
                "age_group": age_group,
                "sex": sex,
                "site": site,
                "country": country,
                "region": region,
                "nutri_counsel_disch": nutri_counsel,
                "poor_feeding": poor_feeding,
                "acutely_ill": acutely_ill,
                "binary_treatment": binary_treatment,
                "pscore": pscore,
                "iptw": iptw,
                "onoutpfeed_prog": onoutpfeed_prog,
                "missed_visit": missed_visit,
                "censored": censored_flag,
                "timepoint_censored": timepoint_censored,
                "ite_weight": ite_weight,
                "ite_feed": ite_feed,
                "pct_weight_gain_factual": pct_weight_gain_factual,
                "pct_weight_gain_counterfactual": pct_weight_gain_counterfactual,
                "feed_outcome_factual": feed_outcome_factual,
                "feed_outcome_counterfactual": feed_outcome_counterfactual
            })

    return pd.DataFrame(data)
