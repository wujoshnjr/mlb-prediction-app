# Changelog

## Unreleased

### Productization Mega Update v6

Added productization scaffolding:

- Baseline comparison report
- CLV slice reports
- Calibration report
- Walk-forward evaluation scaffold
- Static HTML dashboard
- Pytest test suite scaffold
- Documentation package

Strengthened governance:

- Paper-only governance
- Data quality gate
- Training sample gate
- Report health gate
- Feature governance
- Tracking-only feature separation
- Missing data availability flags

Known status:

- Live betting remains disabled.
- ML production loading remains blocked while clean sample count is below 300.
- Calibration is not ready until at least 500 settled predictions.
- Walk-forward validation is scaffolded but not yet sufficient for promotion.
