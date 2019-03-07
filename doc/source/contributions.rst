Contribute to VDRP
******************

How To
======

The suggested workflow for implementing bug fixes and/or new features is the
following:

* Identify or, if necessary, add to our `redmine issue tracker
  <https://luna.mpe.mpg.de/redmine/projects/vdrp>`_ one or more issues
  to tackle. Multiple issues can be addressed together if they belong together.
  Assign the issues to yourself.
* Create a new branch from the trunk with a name either referring to the topic
  or the issue to solve. E.g. if you need to add a new executable, tracked by
  issue #1111
  ``do_something``::

    svn cp ^/trunk ^/branches/do_something_1111\
    -m 'create branch to solve issue #1111'

* Switch to the branch::

    svn switch ^/branches/do_something_1111

* Implement the required changes and don't forget to track your progress on
  redmine. If the feature/bug fix requires a large amount of time, we suggest,
  when possible, to avoid one big commit at the end in favour of smaller
  commits. In this way, in case of breakages, is easier to traverse the branch
  history and find the offending code. For each commit you should add an entry
  in the ``Changelog`` file.

  If you work on multiple issues on the same branch, close one issue before
  proceeding to the next. When closing one issue is good habit to add in the
  description on the redmine the revision that resolves it.
* Every function or class added or modified should be adequately documented as
  described in :ref:`code_style`.

  Documentation is essential both for users and for your fellow developers to
  understand the scope and signature of functions and classes. If a new module
  is added, it should be also added to the documentation in the appropriate
  place. See the existing documentation for examples.

  Each executable should be documented and its description should contain
  enough information and examples to allow users to easily run it.
* Every functionality should be thoroughly tested for python 3.5 or 3.6
  in order to ensure that the code behaves as expected and that future
  modifications will not break existing functionalities. When fixing bugs, add
  tests to ensure that the bug will not repeat. For more information see
  :ref:`testing`.
* Once the issue(s) are solved and the branch is ready, merge any pending change
  **from** the trunk::

    svn merge ^/trunk

  While doing the merge, you might be asked to manually resolve one or more
  conflicts.  Once all the conflicts have been solved, commit the changes with a
  meaningful commit message, e.g.: ``merge ^/trunk into
  ^/branches/do_something_1111``.  Then rerun the test suite to make sure your
  changes do not break functionalities implemented while you were working on
  your branch.
* Then contact the maintainer of ``fplaneserver`` and ask to merge your branch **back
  to the trunk**.

Information about branching and merging can be found in the `svn book
<http://svnbook.red-bean.com/en/1.8/svn.branchmerge.html>`_. For any questions or
if you need support do not hesitate to contact the maintainer or the other
developers.

.. _code_style:

Coding style
============

All the code should be compliant with the official python style guidelines
described in :pep:`8`. To help you keep the code in spec, we suggest to install
plugins that check the code for you, like `Synstastic
<https://github.com/scrooloose/syntastic>`_ for vim or `flycheck
<http://www.flycheck.org/en/latest/>`_ for Emacs.

The code should also be thoroughly documented using the `numpy style
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_. See
the existing documentation for examples.

.. _testing:

Testing
=======

.. note::
    Every part of the code should be tested and should run at least under python
    3.5 and possibly 3.6

``fplaneserver`` uses the testing framework provided by the `robot framework package
<https://robotframework.org>`_. The tests should cover every
aspect of a function or method. If exceptions are explicitly raised, this should
also tested to ensure that the implementation behaves as expected.

The preferred way to run the tests is using `tox
<https://testrun.org/tox/latest/index.html>`_, an automatised test help
package. If you have installed tox, with e.g. ``pip install tox``, you can run
it by typing::

    tox

It will take care of creating virtual environments for every supported version
of python, if it exists on the system, install ``fplaneserver``, its dependences and the
packages necessary to run the tests and runs ``py.test``

You can run the tests for a specific python version using::

    python -m robot

A code coverage report is also created and can be visualized opening
into a browser ``cover/index.html``. 
    
Besides running the tests, the ``tox`` command also builds, by default, the
documentation and collates the coverage tests from the various python
interpreters and can copy then to some directory. To do the latter create, if
necessary, the configuration file ``~/.config/little_deploy.cfg`` and add to it
a section called ``fplaneserver`` with either one or both of the following options:

.. code-block:: ini

    [fplaneserver]
    # if given the deploys the documentation to the given dir
    doc = /path/to/dir
    # if given the deploys the coverage report to the given dir
    cover = /path/to/other/dir

    # it's also possible to insert the project name and the type of the document
    # to deploy using the {project} and {type_} placeholders. E.g
    # cover = /path/to/dir/{project}_{type_}
    # will be expanded to /path/to/dir/fplaneserver_cover

For more information about the configuration file check `little_deploy
<https://github.com/montefra/little_deploy>`_. 


Documentation
=============

To build the documentation you need the additional dependences described in
:ref:`pydep`. They can be installed by hand or during ``fplaneserver`` installation
by executing one of the following commands on a local copy::

  pip install /path/to/fplaneserver[doc]
  pip install /path/to/fplaneserver[livedoc]

The first install ``sphinx``, the ``alabaster`` theme and the ``numpydoc``
extension; the second also installs ``sphinx-autobuild``.

To build the documentation in html format go to the ``doc`` directory and run::

  make html

The output is saved in ``_doc/build/html``. For the full list of available
targets type ``make help``.

If you are updating the documentation and want avoid the
``edit-compile-browser refresh`` cycle, and you have installed
``sphinx-autobuild``, type::

  make livehtml

then visit http://127.0.0.1:8000. The html documentation is automatically
rebuilt after every change of the source and the browser reloaded.

Please make sure that every module in ``fplaneserver`` is present in the
:doc:`codedoc/index`.
