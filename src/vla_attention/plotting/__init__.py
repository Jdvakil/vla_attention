"""Paper figure generators.

Each ``figN_*`` module produces one of the 7 figures listed in Section 7
of the project README. All figures share a single matplotlib style defined
in ``plotting.style``.
"""

from .style import apply_style  # noqa: F401
from .fig1_teaser import plot_teaser  # noqa: F401
from .fig2_modality_heatmap import plot_modality_heatmap  # noqa: F401
from .fig3_head_taxonomy import plot_head_taxonomy  # noqa: F401
from .fig4_attention_rollout import plot_attention_rollout  # noqa: F401
from .fig5_visual_ablation import plot_visual_ablation  # noqa: F401
from .fig6_head_knockout import plot_head_knockout  # noqa: F401
from .fig7_data_efficiency import plot_data_efficiency  # noqa: F401
from .fig_bonus_logit_lens import plot_logit_lens  # noqa: F401
from .fig_bonus_activation_patching import plot_activation_patching  # noqa: F401
