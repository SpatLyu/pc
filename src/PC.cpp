#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <numeric>
#include <algorithm>
#include "pc.h"

// Wrapper function to perform pattern causality analysis
// [[Rcpp::export(rng = false)]]
Rcpp::List RcppPC(
    const Rcpp::NumericVector& target,
    const Rcpp::NumericVector& source,
    const Rcpp::IntegerVector& lib,
    const Rcpp::IntegerVector& pred,
    const Rcpp::IntegerVector& E,
    const Rcpp::IntegerVector& tau,
    int style = 0,
    int num_neighbors = 4,
    int zero_tolerance = 0,
    const std::string& dist_metric = "euclidean",
    bool relative = true,
    bool weighted = true,
    int threads = 1,
    int h = 0,
    Rcpp::Nullable<Rcpp::List> nb = R_NilValue,
    Rcpp::Nullable<int> nrows = R_NilValue)
{
    // --- Input Conversion and Validation --------------------------------------

    std::vector<double> tg = Rcpp::as<std::vector<double>>(target);
    std::vector<double> sg = Rcpp::as<std::vector<double>>(source);
    const size_t n_obs = tg.size();

    // Convert library indices (R 1-based → C++ 0-based)
    const size_t n_lib = static_cast<size_t>(lib.size());
    std::vector<size_t> lib_std;
    lib_std.reserve(n_lib);
    for (size_t i = 0; i < n_lib; ++i) 
    {
        if (lib[i] < 1 || lib[i] > n_obs)
            Rcpp::stop("lib contains out-of-bounds index at position %d (value: %d)", 
                        static_cast<int>(i + 1), lib[i]);
        if (!std::isnan(tg[lib[i] - 1]) && !std::isnan(sg[lib[i] - 1]))
            lib_std.push_back(static_cast<size_t>(lib[i] - 1));
    }

    // Convert prediction indices (R 1-based → C++ 0-based)
    const size_t n_pred = static_cast<size_t>(n_pred.size());
    std::vector<size_t> pred_std;
    pred_std.reserve(n_pred);
    for (int i = 0; i < n_pred; ++i) 
    {
        if (pred[i] < 1 || pred[i] > n_obs)
            Rcpp::stop("pred contains out-of-bounds index at position %d (value: %d)", 
                        static_cast<int>(i + 1), pred[i]);
        if (!std::isnan(tg[pred[i] - 1]) && !std::isnan(sg[pred[i] - 1]))
            pred_std.push_back(static_cast<size_t>(pred[i] - 1));
    }

    // Convert Rcpp IntegerVector to std::vector<size_t>
    std::vector<size_t> E_vec = Rcpp::as<std::vector<size_t>>(E);
    std::vector<size_t> tau_vec = Rcpp::as<std::vector<size_t>>(tau);

    // Expand E and tau
    std::vector<size_t> E_std(2);
    std::vector<size_t> tau_std(2);

    // ---- E ----
    if (E_vec.size() == 1) 
    {
        std::fill(E_std.begin(), E_std.end(), E_vec[0]);
    } 
    else 
    {
        E_std[0] = E_vec[0];
        E_std[1] = E_vec[1];
    }

    // ---- tau ----
    if (method_vec.size() == 1) 
    {
        std::fill(method_expanded.begin(), method_expanded.end(), method_vec[0]);
    } 
    else if (method_vec.size() == 2) 
    {
        method_expanded[0] = method_vec[0];
        std::fill(method_expanded.begin() + 1, method_expanded.end(), method_vec[1]);
    } 
    else 
    {
        method_expanded[0] = method_vec[0];
        for (size_t i = 1; i < nag_raw + 1; ++i) 
        {
            size_t idx = (i - 1) % (method_vec.size() - 1);
            method_expanded[i] = method_vec[idx + 1];
        }
    }

    // Create ordering index
    std::vector<size_t> order(ag_raw.size(), 0);
    std::iota(order.begin(), order.end(), 0);

    // Sort indices by ag_raw
    std::sort(order.begin(), order.end(),
            [&](size_t i, size_t j) {
                return ag_raw[i] < ag_raw[j];
            });

    // Apply ordering
    std::vector<size_t> ag_sorted;
    std::vector<size_t> bin_sorted;
    std::vector<std::string> method_sorted;

    for (size_t idx : order) 
    {
        ag_sorted.push_back(ag_raw[idx]);
        bin_sorted.push_back(bin_expanded[idx + 1]);
        method_sorted.push_back(method_expanded[idx + 1]);
    }

    // Deduplicate while keeping alignment
    std::vector<size_t> ag;
    std::vector<size_t> bin_final;
    std::vector<std::string> method_final;

    for (size_t i = 0; i < ag_sorted.size(); ++i) 
    {
        if (i == 0 || ag_sorted[i] != ag_sorted[i - 1]) 
        {
            ag.push_back(ag_sorted[i]);
            bin_final.push_back(bin_sorted[i]);
            method_final.push_back(method_sorted[i]);
        }
    }

    if (ag.empty())
        Rcpp::stop("Agent indices should not be empty");

    std::vector<size_t> E_std = Rcpp::as<std::vector<size_t>>(E);
    std::vector<size_t> tau_std = Rcpp::as<std::vector<size_t>>(tau);

    // --- Embedding Construction ------------------------------------------------
    std::vector<std::vector<double>> Mx;
    std::vector<std::vector<double>> My;

    if (nb.isNotNull()) 
    {
        // Convert Rcpp::List to std::vector<std::vector<size_t>>
        std::vector<std::vector<size_t>> nb_std = pc::convert::nb2std(nb.get());
        Mx = pc::embed::embed(
            tg, nb_std, E_std[0], tau_std[0], static_cast<size_t>(std::abs(style)));
        My = pc::embed::embed(
            sg, nb_std, E_std[1], tau_std[1], static_cast<size_t>(std::abs(style)));
    } 
    else if (nrows.isNotNull())
    {   
        size_t n_rows = static_cast<size_t>(std::abs(Rcpp::as<int>(nrows)));

        std::vector<std::vector<double>> tm = 
            pc::embed::gridVec2Mat(tg, n_rows);
        Mx = pc::embed::embed(
            tm, E_std[0], tau_std[0], static_cast<size_t>(std::abs(style)));

        std::vector<std::vector<double>> sm = 
            pc::embed::gridVec2Mat(sg, n_rows);
        My = pc::embed::embed(
            sm, E_std[1], tau_std[1], static_cast<size_t>(std::abs(style)));
    }
    else  
    {
        Mx = pc::embed::embed(
            tg, E_std[0], tau_std[0], static_cast<size_t>(std::abs(style)));
        My = pc::embed::embed(
            sg, E_std[1], tau_std[1], static_cast<size_t>(std::abs(style)));
    }

    // --- Perform Pattern Causality Analysis -------------------------
    pc::patcaus::PatternCausalityRes res = pc::patcaus::patcaus(
        Mx, My, lib_std, pred_std, 
        static_cast<size_t>(std::abs(num_neighbors)),
        static_cast<size_t>(std::abs(zero_tolerance)),
        static_cast<size_t>(std::abs(h)),
        dist_metric, relative, weighted,
        static_cast<size_t>(std::abs(threads)));

    // --- Create DataFrame for per-sample causality ----------------------------

    const size_t n_samples = res.NoCausality.size();
    Rcpp::LogicalVector real_loop(n_samples, false);
    Rcpp::CharacterVector pattern_labels(n_samples, "no");

    for (size_t rl = 0; rl < res.RealLoop.size(); ++rl) 
    {
        size_t idx = res.RealLoop[rl];
        if (idx < n_samples) 
        {
            // Record validated samples
            real_loop[idx] = true;
            // Map pattern_types (0–3) → descriptive string labels
            switch (res.PatternTypes[rl]) 
            {
                case 0: pattern_labels[idx]  = "no"; break;
                case 1: pattern_labels[idx]  = "positive"; break;
                case 2: pattern_labels[idx]  = "negative"; break;
                case 3: pattern_labels[idx]  = "dark"; break;
                default: pattern_labels[idx] = "unknown"; break;
            }
        }
    }

    Rcpp::DataFrame causality_df = Rcpp::DataFrame::create(
        Rcpp::Named("no") = Rcpp::NumericVector(res.NoCausality.begin(), res.NoCausality.end()),
        Rcpp::Named("positive") = Rcpp::NumericVector(res.PositiveCausality.begin(), res.PositiveCausality.end()),
        Rcpp::Named("negative") = Rcpp::NumericVector(res.NegativeCausality.begin(), res.NegativeCausality.end()),
        Rcpp::Named("dark") = Rcpp::NumericVector(res.DarkCausality.begin(), res.DarkCausality.end()),
        Rcpp::Named("type") = pattern_labels,
        Rcpp::Named("valid") = real_loop
    );

    // --- Create summary DataFrame for causal strengths ------------------------

    Rcpp::CharacterVector causal_type = Rcpp::CharacterVector::create("positive", "negative", "dark");
    Rcpp::NumericVector causal_strength = Rcpp::NumericVector::create(res.TotalPos, res.TotalNeg, res.TotalDark);

    Rcpp::DataFrame summary_df = Rcpp::DataFrame::create(
        Rcpp::Named("type") = causal_type,
        Rcpp::Named("strength") = causal_strength
    );

    // --- Return structured results --------------------------------------------

    return Rcpp::List::create(
        Rcpp::Named("causality") = causality_df,
        Rcpp::Named("summary") = summary_df
    );
}
