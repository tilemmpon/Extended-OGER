from distutils.core import setup

short_description = "Oger Toolbox"

long_description = """
Oger is a Python toolbox for rapidly building, training and evaluating hierarchical learning architectures on large datasets. It builds functionality on top of the Modular toolkit for Data Processing (MDP). Oger builds functionality on top of MDP, such as:
 - Cross-validation of datasets
 - Grid-searching large parameter spaces
 - Processing of temporal datasets
 - Gradient-based training of deep learning architectures
 - Interface to the Speech Processing, Recognition, and Automatic Annotation Kit (SPRAAK) 

 In addition, several additional MDP nodes are provided by the Oger, such as a:
 - Reservoir node
 - Extended reservoir node with input to output mapping
 - Leaky reservoir node
 - Ridge regression node
 - Conditional Restricted Boltzmann Machine (CRBM) node
 - Delay line reservoir (DLR) node
 - Delay line with feedback reservoir (DLRB) node
 - Simple cycle reservoir (SCR) node
 - Cycle reservoir with jumps (CRJ) node
 - Feed forward ESN (FF-ESN) reservoir node
 - Sparse and orthogonal matrices reservoir (SORM) node
 - Cyclic SORMs reservoir (CyclicSORM) node

To the original Oger many new ESN models presented in the literature have been added by Tilemachos Bontzorlos in a R&D project in Master of Autonomous Systems in Bonn-Rhein-Sieg University of Applied Sciences.
"""

classifiers = ["Intended Audience :: Developers",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]


setup(name = 'Oger', version = '1.2',
      author = "Philemon Brakel, Martin Fiers, Sander Dieleman, Fiontann O'Donnell, Benjamin Schrauwen, David Verstraeten and Tilemachos Bontzorlos",
      author_email = 'pbpop3@gmail.com, mfiers@intec.ugent.be, sander.dieleman@elis.ugent.be, fodonnel@elis.ugent.be, benjamin.schrauwen@elis.ugent.be, david.verstraeten@elis.ugent.be, tilemmpon@hotmail.com',
      maintainer = "Tilemachos Bontzorlos",
      maintainer_email = 'tilemmpon@hotmail.com',
      platforms = ["Any"],
      url = 'http://organic.elis.ugent.be',
      download_url = '',
      description = short_description,
      long_description = long_description,
      classifiers = classifiers,
      packages = ['Oger', 'Oger.datasets', 'Oger.evaluation', 'Oger.gradient', 'Oger.utils', 'Oger.nodes', 'Oger.parallel'],
      package_data={'Oger': ['examples/*.py', 'examples/java_python_interface/*']},
      )
