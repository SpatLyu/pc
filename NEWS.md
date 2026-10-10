# pc 0.3

## new

* Provide `dmi` generic for *delayed mutual information* method (#77).

### enhancements

* Clarify output indexing and boundary handling in `fnn` generic (#69).

### breaking changes

* Euclidean/Manhattan distances now automatically compensate for dimensions skipped due to `NA/NaN`, aligned with base R `dist()` function (#79).

### bug fixes

* Fix incorrect prediction horizon parameter in the time series pattern causality implementation (#80).

# pc 0.2

### new

* Document and maintain built-in case datasets for reproducible analysis (#61).

* Extend `pc` generic with visualization capabilities (#55).

* Provide `fnn` generic for *false nearest neighbors* method (#49).

* Support linear trend removal for spatial cross-sectional data (#36).

### enhancements

* Support sampling with and without replacement in the bootstrapped version of pattern causality (#58).

* Enable method dispatch compatibility in `pc` and `ops` generics via `...` (#43).

* Improve handling of large-scale inputs (#33).

### bug fixes

* Fix unintended CCM-style xmap behavior in pattern causality estimation (#52).

# pc 0.1

* First stable release (#24).
