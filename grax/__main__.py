import huf.cli
from absl import app

import grax.config  # pylint: disable=unused-import
import grax.huf_utils  # pylint: disable=unused-import

if __name__ == "__main__":
    app.run(huf.cli.app_main)
