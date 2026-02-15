import sys
import traceback


class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception = None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message, error_detail):
        if error_detail is not None:
            tb = error_detail.__traceback__
            while tb.tb_next:
                tb = tb.tb_next

            file_name = tb.tb_frame.f_code.co_filename
            line_number = tb.tb_lineno
            error_msg = str(error_detail)
        else:
            file_name = "Unknown"
            line_number = "Unknown"
            error_msg = "No underlying exception"

        return (
            f"{message} | "
            f"Error : {error_msg} | "
            f"File Name : {file_name} | "
            f"Line Number : {line_number}"
        )

    def __str__(self):
        return self.error_message
