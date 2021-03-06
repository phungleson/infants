require 'csv'
require 'json'

filenames = [
  "linkco2007us_den.csv",
  "linkco2008us_den.csv",
  "linkco2009us_den.csv",
  "linkco2010us_den.csv",
]

births_columns_numbers = {}
CSV.foreach('births_columns_numbers.csv') do |row|
  column_name, columns_numbers = row
  births_columns_numbers[column_name] = JSON.parse(columns_numbers)
end

births_columns_categories = {}
CSV.foreach('births_columns_categories.csv') do |row|
  column_name, column_categories = row
  births_columns_categories[column_name] = JSON.parse(column_categories)
end

births_columns_nans = {}
CSV.foreach('births_columns_nans.csv') do |row|
  column_name, column_nan = row
  births_columns_nans[column_name] = JSON.parse(column_nan)
end

csv_out = CSV.open('births_scales.csv', 'w')
all_columns = []
columns = []

filenames.each do |filename|
  linenumber = -1
  new_all_columns = []
  new_columns = []

  CSV.foreach(filename) do |values|
    linenumber += 1

    if linenumber % 10_000 == 0
      puts "Processing filename=#{filename},linenumber=#{linenumber}"
    end

    if linenumber == 0
      new_all_columns = values
      new_columns = new_all_columns - births_columns_nans.keys

      # ignore nan column
      if columns.size == 0
        all_columns = new_all_columns
        columns = new_columns
        csv_out << columns
      else
        columns.each_with_index do |e, index|
          new_index = new_columns.index(e)
          if index != new_index
            raise 'Headers are not the same'
          end
        end
      end

      next
    end

    scaled_values_hash = {}
    nan_count = 0
    values.each_with_index do |value, index|
      column_name = all_columns[index]

      # ignore nan column
      scaled_value = if births_columns_nans.keys.include?(column_name)
        nan_count += 1
        nil
      end

      if births_columns_numbers[column_name]
        column_mumbers = births_columns_numbers[column_name]
        # mean is the default vaule for nan
        scaled_value = column_mumbers['mean']

        if value != ''
          min = column_mumbers['min']
          max = column_mumbers['max']
          scaled_value = (value.to_f - min) / (max - min)
        end
      end

      if births_columns_categories[column_name]
        columns_categories = births_columns_categories[column_name]
        scaled_value = columns_categories[value || '']
      end

      # puts "column_name=#{column_name}, value=#{value}, scaled_value=#{scaled_value || 'empty'}"
      # puts "numbers=#{!births_columns_numbers[column_name].nil?}, categories=#{!births_columns_categories[column_name].nil?}"

      scaled_values_hash[column_name] = scaled_value
    end

    # ignore nan column
    scaled_values_hash = scaled_values_hash.select do |k, v|
      !v.nil?
    end

    scaled_values = scaled_values_hash.sort_by do |key, value|
      columns.index(key)
    end.map do |key, value|
      value
    end

    if scaled_values.size == columns.size
      csv_out << scaled_values
    else
      raise "Something wrong linenumber=#{linenumber} scaled_values.size=#{scaled_values.size}, columns.size=#{columns.size}, nan_count=#{nan_count}"
    end
  end
end
