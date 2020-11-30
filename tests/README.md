# Tests

## Environment tests

In the folder 'environment_tests' are all tests for the project itself.
<br />For example all the tests for handlers, functions and commands.
<br />Every testable entity has a subfolder.
<br />Tests are made with the built-in unittest package and they can be called from teh projects root folder with:<br /> `python -m unittest discover tests\environment_tests\`<br />This will call all the tests it can find from every subfolder.<br />To call tests for only one entity add the subfolder to the call:<br />`python -m unittest discover tests\environment_tests\dataset_handler\`<br />And to call only tests for a specific testset: `python -m unittest tests.environment_tests.dataset_handler.test_mnist` for example.
