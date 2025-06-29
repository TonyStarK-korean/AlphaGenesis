class Validators:
    @staticmethod
    def is_valid_dataframe(df, required_columns=None):
        if required_columns is None:
            required_columns = []
        if df is None:
            return False
        for col in required_columns:
            if col not in df.columns:
                return False
        return True

    @staticmethod
    def is_positive_number(value):
        try:
            return float(value) > 0
        except Exception:
            return False 