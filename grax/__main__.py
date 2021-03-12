from absl import app

import grax.config  # pylint: disable=unused-import
import grax.huf_utils  # pylint: disable=unused-import
import huf.cli

if __name__ == "__main__":
    app.run(huf.cli.app_main)
