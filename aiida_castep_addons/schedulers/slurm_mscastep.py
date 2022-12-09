"""
Scheduler module for MS CASTEP
"""
from aiida.schedulers.plugins.slurm import SlurmScheduler
from aiida.common.escaping import escape_for_bash


class MSCastepSlurmScheduler(SlurmScheduler):
    """
    Special SLURM scheduler for MS CASTEP

    """

    def _get_run_line(self, codes_info, codes_run_mode):
        # Alter the run line
        from aiida.common.datastructures import CodeRunMode

        list_of_runlines = []

        for code_info in codes_info:
            computer_use_double_quotes = code_info.use_double_quotes[0]
            code_use_double_quotes = code_info.use_double_quotes[1]

            command_to_exec_list = []
            for arg in code_info.cmdline_params:
                if arg == "aiida":
                    oldarg = arg
                    for arg in code_info.prepend_cmdline_params:
                        command_to_exec_list.append(
                            escape_for_bash(
                                arg, use_double_quotes=computer_use_double_quotes
                            )
                        )
                    command_to_exec_list.append(
                        escape_for_bash(
                            oldarg, use_double_quotes=code_use_double_quotes
                        )
                    )
                else:
                    command_to_exec_list.append(
                        escape_for_bash(arg, use_double_quotes=code_use_double_quotes)
                    )

            command_to_exec = " ".join(command_to_exec_list)

            escape_stdin_name = escape_for_bash(
                code_info.stdin_name, use_double_quotes=computer_use_double_quotes
            )
            escape_stdout_name = escape_for_bash(
                code_info.stdout_name, use_double_quotes=computer_use_double_quotes
            )
            escape_sterr_name = escape_for_bash(
                code_info.stderr_name, use_double_quotes=computer_use_double_quotes
            )

            stdin_str = f"< {escape_stdin_name}" if code_info.stdin_name else ""
            stdout_str = f"> {escape_stdout_name}" if code_info.stdout_name else ""

            join_files = code_info.join_files
            if join_files:
                stderr_str = "2>&1"
            else:
                stderr_str = f"2> {escape_sterr_name}" if code_info.stderr_name else ""

            output_string = f"{command_to_exec} {stdin_str} {stdout_str} {stderr_str}"

            list_of_runlines.append(output_string)

        self.logger.debug(f"_get_run_line output: {list_of_runlines}")

        if codes_run_mode == CodeRunMode.PARALLEL:
            list_of_runlines.append("wait\n")
            return " &\n\n".join(list_of_runlines)

        if codes_run_mode == CodeRunMode.SERIAL:
            return "\n\n".join(list_of_runlines)

        raise NotImplementedError("Unrecognized code run mode")
