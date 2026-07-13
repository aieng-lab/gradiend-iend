"""Tests for matplotlib plot style configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gradiend.visualizer.plot_style import (
    ENV_FONT_FAMILY,
    ENV_FONT_PATH,
    ENV_LATEX_PREAMBLE_EXTRA,
    ENV_TRANSITION_ARROWS,
    ENV_USE_LATEX,
    check_plot_environment,
    configure_matplotlib_style,
    configure_plot_style,
    reset_matplotlib_style_config,
)
from gradiend.visualizer.plot_style_config import PlotStyleConfig, reset_active_plot_style


@pytest.fixture(autouse=True)
def _reset_plot_style():
    reset_matplotlib_style_config()
    reset_active_plot_style()
    yield
    reset_matplotlib_style_config()
    reset_active_plot_style()


@pytest.fixture
def mpl_rc():
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    original = mpl.rcParams.copy()
    yield mpl.rcParams
    mpl.rcParams.update(original)


class TestPlotStyle:
    def test_latex_probe_covers_largest_automatic_group_label_size(self, monkeypatch):
        pytest.importorskip("matplotlib")
        import matplotlib.text as mtext
        import gradiend.visualizer.plot_style as plot_style_module

        observed_font_sizes = []

        def _reject_missing_14pt_font(fig, *_args, **_kwargs):
            observed_font_sizes.extend(
                text.get_fontsize() for text in fig.findobj(mtext.Text)
            )
            if 14.0 in observed_font_sizes:
                raise LookupError("missing tcss1440")

        monkeypatch.setattr(plot_style_module, "_latex_on_path", lambda: True)
        monkeypatch.setattr("matplotlib.figure.Figure.savefig", _reject_missing_14pt_font)

        assert plot_style_module._latex_usable() is False
        assert 14.0 in observed_font_sizes

    def test_auto_disables_usetex_when_latex_missing(self, mpl_rc, monkeypatch):
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        monkeypatch.delenv(ENV_FONT_PATH, raising=False)
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=False):
            configure_plot_style(force=True)
        assert mpl_rc["text.usetex"] is False

    def test_auto_enables_usetex_when_latex_available(self, mpl_rc, monkeypatch):
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_plot_style(force=True)
        assert mpl_rc["text.usetex"] is True
        assert mpl_rc["font.family"] in ("serif", ["serif"])

    def test_usetex_keeps_explicit_non_default_font_family(self, mpl_rc, monkeypatch):
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        mpl_rc["font.family"] = "monospace"
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_plot_style(force=True)
        assert mpl_rc["text.usetex"] is True
        assert mpl_rc["font.family"] in ("monospace", ["monospace"])

    def test_usetex_keeps_explicit_sans_serif_font_family_from_config(self, mpl_rc, monkeypatch):
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_plot_style(
                PlotStyleConfig(use_latex=True, font_family="sans-serif"),
                force=True,
            )
        assert mpl_rc["text.usetex"] is True
        assert mpl_rc["font.family"] in ("sans-serif", ["sans-serif"])

    def test_force_usetex_off(self, mpl_rc, monkeypatch):
        monkeypatch.setenv(ENV_USE_LATEX, "0")
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_plot_style(force=True)
        assert mpl_rc["text.usetex"] is False

    def test_force_usetex_on_when_available(self, mpl_rc, monkeypatch):
        monkeypatch.setenv(ENV_USE_LATEX, "1")
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_plot_style(force=True)
        assert mpl_rc["text.usetex"] is True
        assert "amsmath" in str(mpl_rc["text.latex.preamble"])
        assert "amssymb" in str(mpl_rc["text.latex.preamble"])

    def test_configure_ensures_amssymb_even_when_already_configured(self, mpl_rc, monkeypatch):
        pytest.importorskip("matplotlib")
        import gradiend.visualizer.plot_style as plot_style_module

        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        mpl_rc["text.latex.preamble"] = r"\usepackage{amsmath}"
        plot_style_module._CONFIGURED = True
        mpl_rc["text.usetex"] = True
        configure_plot_style()
        assert "amssymb" in str(mpl_rc["text.latex.preamble"])

    def test_configure_ensures_amsmath_even_when_already_configured(self, mpl_rc, monkeypatch):
        pytest.importorskip("matplotlib")
        import gradiend.visualizer.plot_style as plot_style_module

        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        mpl_rc["text.latex.preamble"] = ""
        plot_style_module._CONFIGURED = True
        mpl_rc["text.usetex"] = True
        configure_plot_style()
        assert "amsmath" in str(mpl_rc["text.latex.preamble"])
        assert "amssymb" in str(mpl_rc["text.latex.preamble"])

    def test_latex_preamble_extra_from_config(self, mpl_rc, monkeypatch):
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_plot_style(
                PlotStyleConfig(use_latex=True, latex_preamble_extra=r"\usepackage{textcomp}"),
                force=True,
            )
        assert "textcomp" in str(mpl_rc["text.latex.preamble"])
        assert "amsmath" in str(mpl_rc["text.latex.preamble"])
        assert "amssymb" in str(mpl_rc["text.latex.preamble"])

    def test_latex_preamble_extra_from_env(self, mpl_rc, monkeypatch):
        monkeypatch.setenv(ENV_USE_LATEX, "1")
        monkeypatch.setenv(ENV_LATEX_PREAMBLE_EXTRA, r"\usepackage{textcomp}")
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_plot_style(force=True)
        assert "textcomp" in str(mpl_rc["text.latex.preamble"])

    def test_configure_matplotlib_style_deprecated(self, mpl_rc, monkeypatch):
        monkeypatch.setenv(ENV_USE_LATEX, "0")
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=False):
            with pytest.warns(DeprecationWarning, match="configure_plot_style"):
                configure_matplotlib_style(force=True)
        assert mpl_rc["text.usetex"] is False

    def test_plot_style_config_from_env_transition_arrows(self, monkeypatch):
        monkeypatch.setenv(ENV_TRANSITION_ARROWS, "ascii")
        config = PlotStyleConfig.from_env()
        assert config.transition_arrows == "ascii"

    def test_plot_style_config_from_env_font_family(self, monkeypatch):
        monkeypatch.setenv(ENV_FONT_FAMILY, "sans-serif")
        config = PlotStyleConfig.from_env()
        assert config.font_family == "sans-serif"

    def test_force_usetex_on_when_unavailable_falls_back(self, mpl_rc, monkeypatch):
        monkeypatch.setenv(ENV_USE_LATEX, "true")
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=False):
            configure_plot_style(force=True)
        assert mpl_rc["text.usetex"] is False

    def test_custom_font_from_env_path(self, mpl_rc, monkeypatch, tmp_path):
        font_path = tmp_path / "DemoFont.ttf"
        font_path.write_bytes(b"font")

        monkeypatch.setenv(ENV_FONT_PATH, str(font_path))
        mock_fp = MagicMock()
        mock_fp.get_name.return_value = "Demo Font"

        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=False), patch(
            "matplotlib.font_manager.fontManager"
        ) as mock_manager, patch(
            "matplotlib.font_manager.FontProperties",
            return_value=mock_fp,
        ):
            configure_plot_style(force=True)

        mock_manager.addfont.assert_called_once_with(str(font_path.resolve()))
        assert mpl_rc["font.family"] in ("Demo Font", ["Demo Font"])
        assert mpl_rc["font.sans-serif"][0] == "Demo Font"

    def test_require_matplotlib_triggers_style_config(self, monkeypatch):
        pytest.importorskip("matplotlib")
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        import gradiend.visualizer.plot_style as plot_style_module
        from gradiend.visualizer.plot_optional import _require_matplotlib

        assert plot_style_module._CONFIGURED is False
        _require_matplotlib()
        assert plot_style_module._CONFIGURED is True

    def test_check_plot_environment_reports_latex_and_font_status(self, monkeypatch, tmp_path, capsys):
        pytest.importorskip("matplotlib")
        font_path = tmp_path / "DemoFont.ttf"
        font_path.write_bytes(b"font")
        monkeypatch.setenv(ENV_USE_LATEX, "1")
        monkeypatch.setenv(ENV_FONT_PATH, str(font_path))
        mock_fp = MagicMock()
        mock_fp.get_name.return_value = "Demo Font"

        with patch(
            "gradiend.visualizer.plot_style._latex_command_paths",
            return_value={"latex": "latex", "pdflatex": None, "xelatex": None, "lualatex": None},
        ), patch("gradiend.visualizer.plot_style._latex_usable", return_value=True), patch(
            "matplotlib.font_manager.fontManager"
        ) as mock_manager, patch(
            "matplotlib.font_manager.FontProperties",
            return_value=mock_fp,
        ):
            configure_plot_style(force=True)
            status = check_plot_environment()

        mock_manager.addfont.assert_any_call(str(font_path.resolve()))
        printed = capsys.readouterr().out
        assert "GRADIEND plot environment: OK" in printed
        assert "LaTeX:" in printed
        assert "Font:" in printed
        assert f"GRADIEND_PLOT_FONT_PATH={font_path}" in printed
        assert "Style:" in printed
        assert "transition_arrows=" in printed
        assert "Matplotlib:" in printed
        assert status["ok"] is True
        assert status["latex"]["preference"] == "force_on"
        assert status["latex"]["on_path"] is True
        assert status["latex"]["usable"] is True
        assert status["latex"]["resolved_text_usetex"] is True
        assert status["font"]["usable"] is True
        assert status["font"]["font_name"] == "Demo Font"
        assert status["matplotlib"]["available"] is True
        assert status["matplotlib"]["style_configured"] is True

    def test_check_plot_environment_warns_for_bad_font_and_forced_latex(self, monkeypatch, capsys):
        pytest.importorskip("matplotlib")
        monkeypatch.setenv(ENV_USE_LATEX, "true")
        monkeypatch.setenv(ENV_FONT_PATH, "missing-font.ttf")

        with patch(
            "gradiend.visualizer.plot_style._latex_command_paths",
            return_value={"latex": None, "pdflatex": None, "xelatex": None, "lualatex": None},
        ):
            status = check_plot_environment()

        printed = capsys.readouterr().out
        assert "GRADIEND plot environment: ISSUES" in printed
        assert "Warnings:" in printed
        assert status["ok"] is False
        assert status["latex"]["preference"] == "force_on"
        assert status["latex"]["on_path"] is False
        assert status["latex"]["usable"] is False
        assert status["font"]["usable"] is False
        assert status["warnings"]

    def test_check_plot_environment_is_publicly_exported(self):
        import gradiend
        import gradiend.visualizer as visualizer

        assert gradiend.check_plot_environment is check_plot_environment
        assert gradiend.configure_plot_style is configure_plot_style
        assert visualizer.check_plot_environment is check_plot_environment
        assert visualizer.configure_plot_style is configure_plot_style

    def test_check_plot_environment_can_suppress_printing(self, monkeypatch, capsys):
        pytest.importorskip("matplotlib")
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        monkeypatch.delenv(ENV_FONT_PATH, raising=False)

        with patch(
            "gradiend.visualizer.plot_style._latex_command_paths",
            return_value={"latex": None, "pdflatex": None, "xelatex": None, "lualatex": None},
        ):
            status = check_plot_environment(print_status=False)

        assert isinstance(status, dict)
        assert capsys.readouterr().out == ""

    def test_check_plot_environment_applies_usetex_by_default(self, mpl_rc, monkeypatch, capsys):
        pytest.importorskip("matplotlib")
        mpl_rc["text.usetex"] = False
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        monkeypatch.delenv(ENV_FONT_PATH, raising=False)

        with patch(
            "gradiend.visualizer.plot_style._latex_command_paths",
            return_value={"latex": "latex", "pdflatex": None, "xelatex": None, "lualatex": None},
        ), patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            status = check_plot_environment()

        printed = capsys.readouterr().out
        assert "GRADIEND plot environment: OK" in printed
        assert "style_configured=True" in printed
        assert status["latex"]["resolved_text_usetex"] is True
        assert status["matplotlib"]["current_text_usetex"] is True
        assert status["matplotlib"]["style_configured"] is True

    def test_check_plot_environment_read_only_reports_style_info(self, mpl_rc, monkeypatch, capsys):
        pytest.importorskip("matplotlib")
        mpl_rc["text.usetex"] = False
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        monkeypatch.delenv(ENV_FONT_PATH, raising=False)

        with patch(
            "gradiend.visualizer.plot_style._latex_command_paths",
            return_value={"latex": "latex", "pdflatex": None, "xelatex": None, "lualatex": None},
        ), patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            status = check_plot_environment(apply_style=False)

        printed = capsys.readouterr().out
        assert "GRADIEND plot environment: OK" in printed
        assert "GRADIEND_PLOT_FONT_PATH=unset" in printed
        assert "Info:" in printed
        assert "plots will set text.usetex=True" in printed
        assert status["ok"] is True
        assert status["latex"]["resolved_text_usetex"] is True
        assert status["matplotlib"]["current_text_usetex"] is False
        assert status["matplotlib"]["style_configured"] is False

    def test_check_plot_environment_warns_when_custom_font_not_applied(
        self, mpl_rc, monkeypatch, tmp_path, capsys
    ):
        pytest.importorskip("matplotlib")
        font_path = tmp_path / "DemoFont.ttf"
        font_path.write_bytes(b"font")
        monkeypatch.setenv(ENV_USE_LATEX, "0")
        monkeypatch.setenv(ENV_FONT_PATH, str(font_path))
        mock_fp = MagicMock()
        mock_fp.get_name.return_value = "Demo Font"

        with patch(
            "matplotlib.font_manager.FontProperties",
            return_value=mock_fp,
        ), patch("matplotlib.font_manager.findfont", return_value="default-font.ttf"):
            status = check_plot_environment(apply_style=False)

        printed = capsys.readouterr().out
        assert "GRADIEND plot environment: OK" in printed
        assert "Info:" in printed
        assert "plots will register GRADIEND_PLOT_FONT_PATH when rendering" in printed
        assert status["ok"] is True
        assert status["font"]["usable"] is True
        assert status["font"]["font_name"] == "Demo Font"
        assert status["matplotlib"]["style_configured"] is False
